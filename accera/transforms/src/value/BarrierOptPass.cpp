////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Chuck Jacobs
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AcceraPasses.h"

#include <ir/include/value/ValueDialect.h>

#include <utilities/include/TypeTraits.h>

#include <mlir/Dialect/Affine/Analysis/LoopAnalysis.h>

#include <mlir/IR/Visitors.h>

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <mlir/Support/FileUtilities.h>

#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/TypeSwitch.h>

#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_os_ostream.h>

#include <algorithm>
#include <memory>
#include <set>
#include <vector>

using namespace mlir;

using namespace accera::ir;
using namespace accera::transforms;
using namespace accera::ir::value;
using namespace accera::utilities;

using ValueBarrierOp = accera::ir::value::BarrierOp;

namespace
{
struct BarrierOptPass : public BarrierOptBase<BarrierOptPass>
{
    BarrierOptPass(bool writeBarrierGraph, std::string barrierGraphFilename)
    {
        this->writeBarrierGraph = writeBarrierGraph;
        this->barrierGraphFilename = barrierGraphFilename;
    }

    enum class MemoryAccessType
    {
        Read,
        Write,
    };

    // Maybe make this a map from op->operands?
    struct MemoryAccessInfo
    {
        Operation* op = nullptr;
        mlir::Value baseMemRef;
        mlir::ValueRange indices;
        mlir::AffineMap accessMap;
        mlir::ValueRange accessMapOperands;
        MemoryAccessType accessType = MemoryAccessType::Read;
        int nodeId = -1;
        // TODO: integer set or something to indicate the subset of the memory being accessed?
        // TODO: loop level?
        // TODO: deal with views
    };

    struct ActiveMemoryState
    {
        std::vector<MemoryAccessInfo> activeReads;
        std::vector<MemoryAccessInfo> activeWrites;

        bool empty() const
        {
            return activeReads.empty() && activeWrites.empty();
        }
    };

    struct BarrierInfo
    {
        ValueBarrierOp barrierOp;
        int weight = 1;
        bool active = true;
        int nodeId = -1;
    };

    void runOnOperation() final;

    void RemoveAllBarriers()
    {
        getOperation()->walk([&](ValueBarrierOp barrier) {
            barrier.erase();
        });
    }

    static ActiveMemoryState Union(const ActiveMemoryState& a, const ActiveMemoryState& b)
    {
        ActiveMemoryState result = a;
        for (const auto& access : b.activeReads)
        {
            if (!Contains(result.activeReads, access))
                result.activeReads.push_back(access);
        }

        for (const auto& access : b.activeWrites)
        {
            if (!Contains(result.activeWrites, access))
                result.activeWrites.push_back(access);
        }

        return result;
    }

    static bool UsesSameMemory(const MemoryAccessInfo& access1, const MemoryAccessInfo& access2)
    {
        return access1.baseMemRef == access2.baseMemRef;
    }

    static bool Contains(const std::vector<MemoryAccessInfo>& activeAccesses, const MemoryAccessInfo& access)
    {
        return std::find_if(activeAccesses.begin(), activeAccesses.end(), [&](const MemoryAccessInfo& activeAccess) {
                   return UsesSameMemory(access, activeAccess);
               }) != activeAccesses.end();
    }

    static bool IsSame(const std::vector<MemoryAccessInfo>& lhs, const std::vector<MemoryAccessInfo>& rhs)
    {
        if (lhs.size() != rhs.size())
            return false;

        return llvm::all_of(lhs, [&](auto& lhsAccess) { return Contains(rhs, lhsAccess); });
    }

    static bool IsSame(const BarrierOptPass::ActiveMemoryState& lhs, const BarrierOptPass::ActiveMemoryState& rhs)
    {
        return IsSame(lhs.activeWrites, rhs.activeWrites) && IsSame(lhs.activeReads, rhs.activeReads);
    }

    class AnalysisNode
    {
    public:
        AnalysisNode() :
            id(nextId++)
        {}

        explicit AnalysisNode(const std::shared_ptr<AnalysisNode>& parent) :
            AnalysisNode()
        {
            AddPredecessor(parent);
        }

        explicit AnalysisNode(const std::vector<std::shared_ptr<AnalysisNode>>& parents) :
            AnalysisNode()
        {
            for (auto parent : parents)
                AddPredecessor(parent);
        }

        AnalysisNode(MemoryAccessInfo memOp, const std::shared_ptr<AnalysisNode>& parent) :
            AnalysisNode(parent)
        {
            memOp.nodeId = id;
            memoryOp = memOp;
        }

        AnalysisNode(BarrierInfo barrier, std::shared_ptr<AnalysisNode>& parent) :
            AnalysisNode(parent)
        {
            barrier.nodeId = id;
            this->barrier = barrier;
        }

        void AddPredecessor(const std::shared_ptr<AnalysisNode>& parent)
        {
            assert(parent);
            prev.emplace_back(parent);
        }

        void RemovePredecessor(const std::shared_ptr<AnalysisNode>& parent)
        {
            assert(parent);
            prev.erase(std::remove(prev.begin(), prev.end(), parent), prev.end());
        }

        void RemovePredecessor(const AnalysisNode& parent)
        {
            prev.erase(std::remove_if(prev.begin(), prev.end(), [&](const std::shared_ptr<AnalysisNode>& n) { return n.get() == &parent; }), prev.end());
        }

        void AddSuccessor(const std::shared_ptr<AnalysisNode>& child)
        {
            next.emplace_back(child);
        }

        void RemoveSuccessor(const std::shared_ptr<AnalysisNode>& child)
        {
            next.erase(std::remove(next.begin(), next.end(), child), next.end());
        }

        void RemoveSuccessor(const AnalysisNode& child)
        {
            next.erase(std::remove_if(next.begin(), next.end(), [&](const std::shared_ptr<AnalysisNode>& n) { return n.get() == &child; }), next.end());
        }

        bool HasConflict() const
        {
            // check for a conflict between reachability and liveAccesses
            for (auto& memOpInfo : reachingDefs.in.activeReads)
            {
                if (Contains(liveAccesses.out.activeWrites, memOpInfo))
                    return true;
            }

            for (auto& memOpInfo : reachingDefs.in.activeWrites)
            {
                if (Contains(liveAccesses.out.activeReads, memOpInfo))
                    return true;
            }

            return false;
        }

        std::vector<std::shared_ptr<AnalysisNode>> prev;
        std::vector<std::shared_ptr<AnalysisNode>> next;
        llvm::Optional<MemoryAccessInfo> memoryOp;
        llvm::Optional<BarrierInfo> barrier;
        std::string tag; // helpful for debugging

        struct DataflowState
        {
            ActiveMemoryState in;
            ActiveMemoryState out;
        };

        DataflowState liveAccesses;
        DataflowState reachingDefs;

        int id;
        static int nextId;
    };

    // TODO: merge const and non-const versions of VisitNodes
    template <typename Fn>
    static void VisitNodes(AnalysisNode& node, std::set<const AnalysisNode*>& seen, Fn visit)
    {
        // If we've already been processed, exit
        if (seen.count(&node) > 0)
            return;

        // Ensure all parents are processed before we process this node
        for (auto& parent : node.prev)
        {
            if (parent->id < node.id && seen.count(parent.get()) == 0)
                return;
        }

        seen.insert(&node);
        visit(node);

        auto nextNodes = node.next;
        std::sort(nextNodes.begin(), nextNodes.end(), [](const std::shared_ptr<AnalysisNode>& a, const std::shared_ptr<AnalysisNode>& b) {
            return a->id < b->id;
        });
        for (auto& child : nextNodes)
        {
            if (seen.find(child.get()) == seen.end())
            {
                VisitNodes(*child, seen, visit);
            }
        }
    }

    template <typename Fn>
    static void VisitNodes(const AnalysisNode& node, std::set<const AnalysisNode*>& seen, Fn visit)
    {
        if (seen.count(&node) > 0)
            return;

        // Ensure all parents are processed before we process this node
        for (auto& parent : node.prev)
        {
            if (parent->id < node.id && seen.count(parent.get()) == 0)
                return;
        }

        seen.insert(&node);
        visit(node);

        auto nextNodes = node.next;
        std::sort(nextNodes.begin(), nextNodes.end(), [](const std::shared_ptr<AnalysisNode>& a, const std::shared_ptr<AnalysisNode>& b) {
            return a->id < b->id;
        });
        for (auto& child : nextNodes)
        {
            if (seen.find(child.get()) == seen.end())
            {
                VisitNodes(*child, seen, visit);
            }
        }
    }

    template <typename Fn>
    static void VisitNodes(AnalysisNode& node, Fn visit)
    {
        std::set<const AnalysisNode*> seen;
        VisitNodes(node, seen, visit);
    }

    template <typename Fn>
    static void VisitNodes(const AnalysisNode& node, Fn visit)
    {
        std::set<const AnalysisNode*> seen;
        VisitNodes(node, seen, visit);
    }

    static void RemoveNode(AnalysisNode& n)
    {
        for (auto& prev : n.prev)
        {
            prev->RemoveSuccessor(n);
            for (auto& succ : n.next)
                prev->AddSuccessor(succ);
        }

        for (auto& next : n.next)
        {
            next->RemovePredecessor(n);
            for (auto& pred : n.prev)
                next->AddPredecessor(pred);
        }
    }

    struct AnalysisGraph
    {
        void Simplify()
        {
            auto shouldDeleteNode = [](const AnalysisNode& n) {
                return (n.prev.size() <= 1 || n.next.size() <= 1) && !n.memoryOp && !n.barrier;
            };

            // remove any node with no memory ops or barriers, unless it's a loop entry or if/else entry
            VisitNodes(*entryNode, [&](AnalysisNode& n) {
                if (shouldDeleteNode(n) && (&n != entryNode.get()) && (&n != exitNode.get()))
                {
                    RemoveNode(n);
                }
            });
        }

        bool Verify() const
        {
            if (!entryNode)
                return true;

            bool valid = true;
            VisitNodes(*entryNode, [&valid](const AnalysisNode& n) {
                // check that all predecessors contain us as a successor
                for (auto& pred : n.prev)
                {
                    if (std::find_if(pred->next.begin(), pred->next.end(), [&n](const std::shared_ptr<AnalysisNode>& x) {
                            return x.get() == &n;
                        }) == pred->next.end())
                    {
                        valid = false;
                    }
                }

                // check that all successors contain us as a predecessor
                for (auto& succ : n.next)
                {
                    if (std::find_if(succ->prev.begin(), succ->prev.end(), [&n](const std::shared_ptr<AnalysisNode>& x) {
                            return x.get() == &n;
                        }) == succ->prev.end())
                    {
                        valid = false;
                    }
                }
            });

            return valid;
        }

        void CollectArrayNames()
        {
            int currentArrayName = 0;
            auto getNextArrayName = [&]() {
                return std::string("A_") + std::to_string(currentArrayName++);
            };

            if (entryNode)
            {
                VisitNodes(*entryNode, [&](AnalysisNode& n) {
                    if (n.memoryOp)
                    {
                        if (arrayNames.count(n.memoryOp->baseMemRef) == 0)
                        {
                            arrayNames[n.memoryOp->baseMemRef] = getNextArrayName();
                        }
                    }
                });
            }
        }

        std::shared_ptr<AnalysisNode> entryNode;
        std::shared_ptr<AnalysisNode> exitNode;
        llvm::DenseMap<mlir::Value, std::string> arrayNames = llvm::DenseMap<mlir::Value, std::string>();

        // Based on algorithm in https://en.wikipedia.org/wiki/Reaching_definition
        void ComputeReachingDefinitions()
        {
            std::vector<AnalysisNode*> worklist;

            // Add all nodes to the worklist and clear their state
            VisitNodes(*entryNode, [&](AnalysisNode& node) {
                node.reachingDefs.in = {};
                node.reachingDefs.out = {};
                worklist.push_back(&node);
            });

            while (!worklist.empty())
            {
                auto node = worklist.back();
                worklist.pop_back();

                node->reachingDefs.in = {};

                // Calculate `in` from predecessors' `out`s
                for (auto& weakPred : node->prev)
                {
                    auto pred = weakPred;
                    for (auto& predRead : pred->reachingDefs.out.activeReads)
                    {
                        if (!Contains(node->reachingDefs.in.activeReads, predRead))
                        {
                            node->reachingDefs.in.activeReads.push_back(predRead);
                        }
                    }

                    for (auto& predWrite : pred->reachingDefs.out.activeWrites)
                    {
                        if (!Contains(node->reachingDefs.in.activeWrites, predWrite))
                        {
                            node->reachingDefs.in.activeWrites.push_back(predWrite);
                        }
                    }
                }

                // Save old out
                auto oldOut = node->reachingDefs.out;
                node->reachingDefs.out = node->reachingDefs.in;

                // out = union(Generated, In-Killed)
                if (node->barrier && node->barrier->active)
                {
                    // kill all reads and writes
                    node->reachingDefs.out = {};
                }

                // visit each memory op
                if (node->memoryOp)
                {
                    if (node->memoryOp->accessType == MemoryAccessType::Write)
                    {
                        if (!Contains(node->reachingDefs.out.activeWrites, *node->memoryOp))
                        {
                            node->reachingDefs.out.activeWrites.push_back(*node->memoryOp);
                        }
                    }
                    else // Read
                    {
                        if (!Contains(node->reachingDefs.out.activeReads, *node->memoryOp))
                        {
                            node->reachingDefs.out.activeReads.push_back(*node->memoryOp);
                        }
                    }
                }

                if (!IsSame(oldOut, node->reachingDefs.out))
                {
                    for (auto& succ : node->next)
                    {
                        worklist.push_back(succ.get());
                    }
                }
            }
        }

        void ComputeLiveness()
        {
            std::vector<AnalysisNode*> worklist;
            worklist.push_back(exitNode.get());

            // Clear in and out states
            VisitNodes(*entryNode, [&](AnalysisNode& node) {
                node.liveAccesses.in = {};
                node.liveAccesses.out = {};
                worklist.push_back(&node);
            });

            while (!worklist.empty())
            {
                auto node = worklist.back();
                worklist.pop_back();

                // out state = union of in-states
                ActiveMemoryState outgoingState;
                for (auto& succ : node->next)
                {
                    outgoingState = Union(outgoingState, succ->liveAccesses.in);
                }

                // incoming state = union(Generated, Out-Killed)
                ActiveMemoryState incomingState = outgoingState;
                if (node->barrier && node->barrier->active)
                    incomingState = {}; // kill all reads and writes

                // visit each memory op
                if (node->memoryOp)
                {
                    if (node->memoryOp->accessType == MemoryAccessType::Write)
                    {
                        if (!Contains(incomingState.activeWrites, *node->memoryOp))
                        {
                            incomingState.activeWrites.push_back(*node->memoryOp);
                        }
                    }
                    else // Read
                    {
                        if (!Contains(incomingState.activeReads, *node->memoryOp))
                        {
                            incomingState.activeReads.push_back(*node->memoryOp);
                        }
                    }
                }

                // If computed in state is different than old in state, add predecessors to worklist
                if (!IsSame(incomingState, node->liveAccesses.in))
                {
                    for (auto& pred : node->prev)
                    {
                        worklist.push_back(pred.get());
                    }
                }

                node->liveAccesses.in = incomingState;
                node->liveAccesses.out = outgoingState;
            }
        }

        void ComputeDataflow()
        {
            ComputeReachingDefinitions();
            ComputeLiveness();
        }

        bool AnalyzeBarriers() const
        {
            bool result = true;
            if (entryNode)
            {
                VisitNodes(*entryNode, [&](AnalysisNode& node) {
                    if (node.barrier && !node.barrier->active && node.HasConflict())
                    {
                        // llvm::errs() << "Conflict at barrier in node " << node.id << "\n";
                        result = false;
                        node.tag += "INVALID<br/>";
                    }
                });
            }

            return result;
        }

        void OptimizeBarriers()
        {
            if (!entryNode)
                return;

            // Get a list of barrier nodes sorted by weight (descending)
            std::vector<AnalysisNode*> barrierNodes;
            VisitNodes(*entryNode, [&](AnalysisNode& node) {
                if (node.barrier)
                    barrierNodes.push_back(&node);
            });

            std::sort(barrierNodes.begin(), barrierNodes.end(), [](const AnalysisNode* a, const AnalysisNode* b) {
                return a->barrier->weight > b->barrier->weight;
            });

            for (auto node : barrierNodes)
            {
                if (node->barrier->active && !node->HasConflict())
                {
                    node->barrier->active = false;

                    // TODO: Can we recompute just the relevant part of the dataflow (e.g.,  by just adding this node to the worklist and iterating?)
                    ComputeDataflow();
                }
            };
        }

        void DeleteInactiveBarrierOps()
        {
            if (entryNode)
            {
                VisitNodes(*entryNode, [&](AnalysisNode& node) {
                    if (node.barrier && !node.barrier->active)
                    {
                        node.barrier->barrierOp->erase();
                        node.barrier->barrierOp = nullptr;
                    }
                });
            }
        }

        void WriteDotFile(llvm::raw_fd_ostream& out) const
        {
            // TODO: add special treatment for entry and exit nodes
            auto getArrayName = [&](mlir::Value& array) -> std::string {
                if (auto it = arrayNames.find(array); it != arrayNames.end())
                    return it->second;
                return "???";
            };

            if (!entryNode)
                return;

            out << "digraph cfg {\n";
            out << "\tlabelloc=\"t\"\n";
            out << "\tlabel=<<b>Barrier Analysis</b>>\n";
            VisitNodes(*entryNode, [&](AnalysisNode& node) {
                SmallString<128> labelString("");
                SmallString<128> nodeAttrString("");
                llvm::raw_svector_ostream labelStream(labelString);
                llvm::raw_svector_ostream nodeAttrStream(nodeAttrString);

                labelStream << "<";
                int penWidth = 2;
                if (&node == entryNode.get())
                {
                    labelStream << "ENTRY<br/>";
                    penWidth = 4;
                }
                else if (&node == exitNode.get())
                {
                    labelStream << "EXIT<br/>";
                    penWidth = 4;
                }

                labelStream << "n" << node.id << "<br/>";

                nodeAttrStream << " penwidth=" << penWidth;
                if (node.barrier)
                {
                    nodeAttrStream << " shape=box";
                    if (!node.barrier->active)
                        labelStream << "<font color=\"gray\"><s>";
                    labelStream << "BARRIER wt " << node.barrier->weight;
                    if (!node.barrier->active)
                        labelStream << "</s></font>";

                    if (node.barrier->active)
                        nodeAttrStream << " color=black";
                    else
                        nodeAttrStream << " color=gray";
                }

                if (node.memoryOp)
                {
                    if (node.memoryOp->accessType == MemoryAccessType::Read)
                    {
                        nodeAttrStream << " shape=box color=firebrick";
                        labelStream << "READ  " << getArrayName(node.memoryOp->baseMemRef);
                    }
                    else
                    {
                        nodeAttrStream << " shape=box color=blue";
                        labelStream << "WRITE " << getArrayName(node.memoryOp->baseMemRef);
                    }
                }

                if (!node.tag.empty())
                {
                    labelStream << "<br/>" << node.tag;
                }

                auto nextNodes = node.next;
                std::sort(nextNodes.begin(), nextNodes.end(), [](const std::shared_ptr<AnalysisNode>& a, const std::shared_ptr<AnalysisNode>& b) {
                    return a->id < b->id;
                });

                labelStream << ">";

                out << "\t" << node.id << " [label=" << labelString << nodeAttrString << "];\n";

                // Write 'next' edges
                for (auto& next : nextNodes)
                {
                    SmallString<128> edgeAttrString("");
                    llvm::raw_svector_ostream edgeAttrStream(edgeAttrString);

                    edgeAttrStream << "[";
                    if (node.id > next->id)
                        edgeAttrStream << " style=dashed";

                    if (!node.reachingDefs.out.empty() || !node.liveAccesses.out.empty())
                    {
                        edgeAttrStream << " label=<<table cellspacing=\"0\" border=\"0\" cellborder=\"1\">";
                        if (!node.reachingDefs.out.empty())
                        {
                            edgeAttrStream << "<tr><td>Reach</td><td>";
                            for (auto& m : node.reachingDefs.out.activeReads)
                            {
                                edgeAttrStream << "READ  " << getArrayName(m.baseMemRef) << " at n" << m.nodeId << "<br/>";
                            }
                            for (auto& m : node.reachingDefs.out.activeWrites)
                            {
                                edgeAttrStream << "WRITE " << getArrayName(m.baseMemRef) << " at n" << m.nodeId << "<br/>";
                            }
                            edgeAttrStream << "</td></tr>";
                        }
                        if (!node.liveAccesses.out.empty())
                        {
                            edgeAttrStream << "<tr><td>Live</td><td>";
                            for (auto& m : node.liveAccesses.out.activeReads)
                            {
                                edgeAttrStream << "READ  " << getArrayName(m.baseMemRef) << " at n" << m.nodeId << "<br/>";
                            }
                            for (auto& m : node.liveAccesses.out.activeWrites)
                            {
                                edgeAttrStream << "WRITE " << getArrayName(m.baseMemRef) << " at n" << m.nodeId << "<br/>";
                            }
                            edgeAttrStream << "</td></tr>]";
                        }
                        edgeAttrStream << "</table>>";
                    }
                    edgeAttrStream << "]";
                    out << "\t" << node.id << " -> " << next->id << ":n " << edgeAttrString << ";\n";
                }
            });
            out << "}\n";
        }
    };

private:
    static mlir::Value GetBaseMemRef(mlir::Value memRef)
    {
        if (!memRef.getDefiningOp())
            return memRef;

        // if a view type, get the base memref from it (recursively)
        if (ViewLikeOpInterface view = dyn_cast<ViewLikeOpInterface>(memRef.getDefiningOp()))
            return GetBaseMemRef(view.getViewSource());

        return memRef;
    }

    static llvm::Optional<MemoryAccessInfo> GetSharedMemoryAccessInfo(Operation* op)
    {
        auto getAccessInfo = [](auto op, MemoryAccessType accessType) -> llvm::Optional<MemoryAccessInfo> {
            auto memRefType = op.getMemRefType();
            auto memSpace = memRefType.getMemorySpaceAsInt();
            if (memSpace == gpu::GPUDialect::getWorkgroupAddressSpace())
            {
                MemoryAccessInfo info;
                info.op = op.getOperation();
                info.baseMemRef = GetBaseMemRef(op.getMemRef());
                info.accessType = accessType;
                return info;
            }
            return llvm::None;
        };

        auto getVectorAccessInfo = [](auto op, MemoryAccessType accessType) -> llvm::Optional<MemoryAccessInfo> {
            auto memRefType = op.getMemRefType();
            auto memSpace = memRefType.getMemorySpaceAsInt();
            if (memSpace == gpu::GPUDialect::getWorkgroupAddressSpace())
            {
                MemoryAccessInfo info;
                info.op = op.getOperation();
                info.baseMemRef = GetBaseMemRef(op.base());
                info.accessType = accessType;
                return info;
            }
            return llvm::None;
        };

        auto getAffineAccessInfo = [getAccessInfo](auto affineOp, MemoryAccessType accessType) -> llvm::Optional<MemoryAccessInfo> {
            if (auto result = getAccessInfo(affineOp, accessType))
            {
                result->accessMap = affineOp.getAffineMap();
                result->accessMapOperands = affineOp.getMapOperands();
                // result->indices = affineOp.getIndices();
                return result;
            }
            return llvm::None;
        };

        auto getBlockLoadAccessInfo = [](GPUBlockCacheOp op, MemoryAccessType accessType) -> llvm::Optional<MemoryAccessInfo> {
            auto memRefType = op.dest().getType().cast<MemRefType>();
            auto memSpace = memRefType.getMemorySpaceAsInt();
            if (memSpace == gpu::GPUDialect::getWorkgroupAddressSpace())
            {
                MemoryAccessInfo info;
                info.op = op.getOperation();
                info.baseMemRef = GetBaseMemRef(op.dest());
                info.accessType = accessType;
                return info;
            }
            return llvm::None;
        };

        return mlir::TypeSwitch<Operation*, llvm::Optional<MemoryAccessInfo>>(op)
            .Case<mlir::AffineReadOpInterface>([&](mlir::AffineReadOpInterface affineLoadOp) {
                return getAffineAccessInfo(affineLoadOp, MemoryAccessType::Read);
            })
            .Case<mlir::AffineWriteOpInterface>([&](mlir::AffineWriteOpInterface affineStoreOp) {
                return getAffineAccessInfo(affineStoreOp, MemoryAccessType::Write);
            })
            .Case<mlir::memref::LoadOp>([&](mlir::memref::LoadOp loadOp) {
                return getAccessInfo(loadOp, MemoryAccessType::Read);
            })
            .Case<mlir::memref::StoreOp>([&](mlir::memref::StoreOp storeOp) {
                return getAccessInfo(storeOp, MemoryAccessType::Write);
            })
            .Case<mlir::vector::LoadOp>([&](mlir::vector::LoadOp loadOp) {
                return getVectorAccessInfo(loadOp, MemoryAccessType::Read);
            })
            .Case<mlir::vector::StoreOp>([&](mlir::vector::StoreOp storeOp) {
                return getVectorAccessInfo(storeOp, MemoryAccessType::Write);
            })
            .Case<GPUBlockCacheOp>([&](GPUBlockCacheOp storeOp) {
                return getBlockLoadAccessInfo(storeOp, MemoryAccessType::Write);
            })
            .Default([](Operation*) { return llvm::None; });

        // TODO:
        //
        // TensorLoadOp -- memref()
        // TensorStoreOp -- memref()
        // PrefetchOp -- memref()
        // AffinePrefetchOp -- memref()
        // TransposeOp -- in()
        // CopyOpInterface -- getSource, getTarget --- ugh: 2 memrefs
    }

    static AnalysisGraph ComputeRegionGraph(mlir::Region& region, const std::shared_ptr<AnalysisNode>& in = {}, int barrierWeight = 1)
    {
        // Create a dedicated entry node for the region
        std::shared_ptr<AnalysisNode> firstNode = std::make_shared<AnalysisNode>();
        if (in)
        {
            firstNode->AddPredecessor(in);
            in->AddSuccessor(firstNode);
        }

        // TODO: move this someplace reasonable
        auto getTripCount = [](mlir::Operation& op, int defaultCount = 1) -> int {
            if (auto affineForOp = llvm::dyn_cast<mlir::AffineForOp>(op))
            {
                return mlir::getConstantTripCount(affineForOp).getValueOr(10);
            }
            else if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(op))
            {
                if (auto lowerBound = forOp.getLowerBound().getDefiningOp<mlir::arith::ConstantOp>())
                {
                    if (auto upperBound = forOp.getUpperBound().getDefiningOp<mlir::arith::ConstantOp>())
                    {
                        if (auto step = forOp.getStep().getDefiningOp<mlir::arith::ConstantOp>())
                        {
                            return (upperBound.getValue().cast<mlir::IntegerAttr>().getInt() - lowerBound.getValue().cast<mlir::IntegerAttr>().getInt()) / step.getValue().cast<mlir::IntegerAttr>().getInt();
                        }
                    }
                }
            }

            return defaultCount;
        };

        std::shared_ptr<AnalysisNode> prevNode = firstNode;
        for (auto& block : region)
        {
            for (auto& op : block)
            {
                if (auto memInfo = GetSharedMemoryAccessInfo(&op))
                {
                    auto node = std::make_shared<AnalysisNode>(*memInfo, prevNode);
                    if (prevNode)
                        prevNode->AddSuccessor(node);
                    prevNode = node;
                }
                else if (auto barrierOp = dyn_cast<ValueBarrierOp>(op))
                {
                    auto node = std::make_shared<AnalysisNode>(BarrierInfo{ { barrierOp }, barrierWeight }, prevNode);
                    if (prevNode)
                        prevNode->AddSuccessor(node);
                    prevNode = node;
                }
                else
                {
                    if (op.getNumRegions() > 0)
                    {
                        bool isLoop = isa<LoopLikeOpInterface>(op) && !op.hasAttr("accv_gpu_map");
                        int loopWeight = 1;
                        if (isLoop)
                            loopWeight *= getTripCount(op, 10);

                        // entry node
                        auto startNode = std::make_shared<AnalysisNode>(prevNode);
                        if (prevNode)
                            prevNode->AddSuccessor(startNode);

                        // parallel regions -- for use when node is an "if"
                        std::vector<std::shared_ptr<AnalysisNode>> regionExitNodes;
                        for (auto& nestedRegion : op.getRegions())
                        {
                            auto graph = ComputeRegionGraph(nestedRegion, startNode, barrierWeight * loopWeight);
                            assert(graph.entryNode);
                            regionExitNodes.emplace_back(graph.exitNode);
                        }

                        // exit node
                        auto exitNode = std::make_shared<AnalysisNode>(regionExitNodes);
                        for (auto& regionExitNode : regionExitNodes)
                        {
                            regionExitNode->AddSuccessor(exitNode);
                        }

                        prevNode = exitNode;

                        // If the op is a loop, we need to add a shortcut (skip) edge and a back edge
                        // Check if the loop is a parallel loop, and set `isLoop` to false in that case
                        if (isLoop)
                        {
                            startNode->tag = "LoopStart<br/>";
                            exitNode->tag = "LoopExit<br/>";

                            // back edge
                            startNode->AddPredecessor(exitNode);
                            exitNode->AddSuccessor(startNode);

                            // skip edge
                            prevNode = startNode;
                        }
                    }
                }
            }
        }

        return { firstNode, prevNode };
    }
};
} // namespace

int BarrierOptPass::AnalysisNode::nextId = 0;

void BarrierOptPass::runOnOperation()
{
    auto op = getOperation();
    auto writeDotFile = [&](auto cfg, std::string filename) {
        cfg.CollectArrayNames();

        if (filename.empty())
        {
            cfg.WriteDotFile(llvm::errs());
        }
        else
        {
            std::string error;
            auto graphDotFile = mlir::openOutputFile(filename, &error);
            if (!graphDotFile)
            {
                op->emitError() << error;
            }
            else
            {
                cfg.WriteDotFile(graphDotFile->os());
                graphDotFile->keep();
            }
        }
    };

    if (op->getNumRegions() != 1)
    {
        op->emitError("Expected a single region");
        return;
    }

    if (auto execTargetOpt = util::ResolveExecutionTarget(op))
    {
        auto execTarget = *execTargetOpt;
        if (execTarget != ExecutionTarget::GPU)
            return;
    }

    if (auto launchAttr = op->getAttrOfType<mlir::ArrayAttr>(ValueFuncOp::getGPULaunchAttrName()))
    {
        auto gpuParams = accera::ir::targets::GPU::FromArrayAttr(launchAttr);
        if (const auto threadsPerWarp = util::ResolveWarpSize(util::ResolveExecutionRuntime(op)))
        {
            auto threadsPerBlock = gpuParams.block.x * gpuParams.block.y * gpuParams.block.z;
            if (threadsPerBlock <= (threadsPerWarp->first * threadsPerWarp->second))
            {
                RemoveAllBarriers();
                return;
            }
        }
    }

    auto cfg = ComputeRegionGraph(op->getRegions()[0]);
    if (cfg.entryNode && cfg.exitNode && (cfg.entryNode != cfg.exitNode))
    {
        assert(cfg.Verify());
        cfg.Simplify();
        assert(cfg.Verify());
        cfg.ComputeDataflow();
        cfg.OptimizeBarriers();

        // Now delete all the inactive barrier ops
        cfg.DeleteInactiveBarrierOps();

        if (writeBarrierGraph)
        {
            cfg.AnalyzeBarriers();
            writeDotFile(cfg, barrierGraphFilename);
        }

        if (!cfg.AnalyzeBarriers())
        {
            op->emitError("Barrier analysis failed");
        }
    }
}

namespace accera::transforms::value
{
std::unique_ptr<mlir::Pass> createBarrierOptPass(bool writeBarrierGraph, std::string barrierGraphFilename)
{
    return std::make_unique<BarrierOptPass>(writeBarrierGraph, barrierGraphFilename);
}

std::unique_ptr<mlir::Pass> createBarrierOptPass()
{
    return std::make_unique<BarrierOptPass>(false, "");
}
} // namespace accera::transforms::value
