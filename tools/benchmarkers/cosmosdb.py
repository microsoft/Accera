####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################
import sys
from azure.cosmos import CosmosClient, PartitionKey
import config
import argparse

HOST = config.settings['host']
MASTER_KEY = config.settings['master_key']
DATABASE_ID = config.settings['database_id']

def get_container(containerName, verboseLogs):
    client = CosmosClient(HOST, credential=MASTER_KEY)
    db = client.get_database_client(DATABASE_ID)

    container = db.create_container_if_not_exists(id=containerName, partition_key=PartitionKey(path='/partitionKey'))
    if verboseLogs:
        print('Fetched container with id \'{0}\''.format(containerName))

    return container

def upsert_benchmark_results(resultRows, containerName, verboseLogs):
    if verboseLogs:
        print(f"Uploading {len(resultRows)} results to Cosmos DB {DATABASE_ID} in container {containerName}...")
    container = get_container(containerName, verboseLogs)

    for row in resultRows:
        container.upsert_item(row)

    if verboseLogs:
        print("Finished Uploading results to Cosmos DB storage.")

def getTopResultFromParition(container, partitionKey):
    topPerf = list(container.query_items(query="SELECT VALUE MAX(c.TFlops) FROM c", partition_key=partitionKey))
    if len(topPerf) == 0:
        return None

    items = list(container.query_items(
        query="SELECT * From c where c.TFlops=@topPerf",
        partition_key=partitionKey,
        parameters=[
            {"name": "@topPerf", "value": topPerf[0]}
        ]))

    return items[0]

def show_benchmark_summary(containerName):
    container = get_container(containerName, False)
    partitions = list(container.query_items(query="SELECT DISTINCT c.partitionKey from c", enable_cross_partition_query=True))
    partitions = list(list(partitionDict.values())[0] for partitionDict in partitions)
    for partitionKey in partitions:
        print(f'Top result for partition {partitionKey}:')
        item = getTopResultFromParition(container, partitionKey)
        if item is None:
            continue

        throughput = item['TFlops']
        mma_shape = item['mma_shape']
        use_static_offsets = item['use_static_offsets']
        cache_layout_A = item['cache_layout_A']
        cache_layout_B = item['cache_layout_B']
        block_tile = item['block_tile']
        k_split = item['k_split']
        double_buffering = item['double_buffering']
        vectorize = item['vectorize']
        num_fused_passes = item['num_fused_passes']
        scheduling_policy = item['scheduling_policy']
        itemId = item['id']
        print(f'{throughput} TFlops')
        print(f'Item id: {itemId}')
        print(f'(Optimizations: MMA shape: {mma_shape}, Static offsets: {use_static_offsets}, CacheA: {cache_layout_A}, CacheB: {cache_layout_B}, Block tile: {block_tile}, k-split: {k_split}, double buffering: {double_buffering}, vectorize: {vectorize}, fused passes: {num_fused_passes}, scheduling policy: {scheduling_policy})')
        print('------------------------------------------')

def main(args=[]):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--container', help='The Cosmos DB container to get summary from', required=True)

    args = parser.parse_args(args)

    show_benchmark_summary(args.container)

if __name__ == "__main__":
    main(sys.argv[1:])