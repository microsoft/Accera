////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Alex Soustek
////////////////////////////////////////////////////////////////////////////////////////////////////

// Load external script from path
function loadScript(scriptUrl) {
    const script = document.createElement('script');
    script.src = scriptUrl;
    document.body.appendChild(script);

    return new Promise((res, rej) => {
        script.onload = function () {
            res();
        }
        script.onerror = function () {
            rej();
        }
    });
}

let plane = null
const shadowMapRes = 8192;
const HTTP_METHODS = { GET: "GET", POST: "POST" };

const AXIS = {
    _0: "_0",
    _1: "_1",
    _2: "_2"
}

const CUBE_SIDES = {
    TOP: "TOP",
    BOTTOM: "BOTTOM",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
    FRONT: "FRONT",
    BACK: "BACK"
}

const LIGHT_TYPES = {
    DIRECTIONAL: "DIRECTIONAL",
    POINT: "POINT"
}

function TweenPosition(three_obj, options) {
    let position = three_obj.position.clone()
    var tween = new TWEEN.Tween(position)
        .to(options.target, options.duration);

    tween.onUpdate(() => {
        three_obj.position.x = position.x;
        three_obj.position.y = position.y;
        three_obj.position.z = position.z;
    });

    return tween;
}

function TweenRotation(three_obj, options) {
    let rotation = three_obj.rotation.clone()
    var tween = new TWEEN.Tween(rotation)
        .to(options.target, options.duration);

    tween.onUpdate(() => {
        three_obj.rotation.x = rotation.x;
        three_obj.rotation.y = rotation.y;
        three_obj.rotation.z = rotation.z;
    });

    return tween;
}

function TweenProperties(options) {
    let initial_props = options.initial;
    var tween = new TWEEN.Tween(initial_props)
        .to(options.target, options.duration);

    tween.onUpdate(() => {
        if (typeof options.onUpdate === 'function') {
            options.onUpdate(initial_props);
        }
    });

    return tween;
}

// Make a reusable function that returns a single Promise
function PerformRequest(method, path, json) {
    json = json ? json : {};
    return new Promise(function (resolve, reject) {
        const getAPIData = new XMLHttpRequest();
        const url = path;
        getAPIData.open(method, url);

        if (method === HTTP_METHODS.POST) {
            getAPIData.setRequestHeader('Content-Type', 'application/json');
            getAPIData.send(JSON.stringify(json));
        }
        else {
            getAPIData.send();
        }

        getAPIData.onload = function () {
            const APIData = JSON.parse(getAPIData.responseText);
            resolve(APIData);
        };
    });
}

function GetBaseSceneObject(viz, parent) {
    let parent_obj = parent ? parent : viz.scene;
    var _root = new THREE.Object3D();
    var obj = {
        parent_object: parent_obj,
        root_object: _root,
        remove: () => {
            if (obj.root_object.parent) {
                obj.root_object.parent.remove(obj.root_object);
            }
        }
    };

    obj.parent_object.add(obj.root_object);
    return obj;
}

function GetDefault3dIterationPose() {
    return { x: 0.65, y: 2.5, z: 3.14 }
}

function GetDefault2dIterationPose() {
    return { x: 0, y: 3.14, z: 3.14 }
}

function GetLayoutInfoOfSpace(space_params) {
    const padding_array = _create_padding_array(space_params.padding);
    const overall_y = (space_params.shape[0] * space_params.size) + ((space_params.shape[0] - 1) * padding_array[0]);
    const overall_x = (space_params.shape[1] * space_params.size) + ((space_params.shape[1] - 1) * padding_array[1]);
    const overall_z = (space_params.shape[2] * space_params.size) + ((space_params.shape[2] - 1) * padding_array[2]);

    const starting_pos = [space_params.position.x - overall_x / 2, space_params.position.y - overall_y / 2, space_params.position.z - overall_z / 2];
    return {
        origin: { x: starting_pos[0], y: starting_pos[1], z: starting_pos[2] },
        center: { x: space_params.position.x, y: space_params.position.y, z: space_params.position.z },
        total_dim_size: { x: overall_x, y: overall_y, z: overall_z }
    }
}

function GenerateIterationSpacePair(viz, options) {
    let obj = GetBaseSceneObject(viz, options.parent)

    const space_1_params = options.space1_params;
    const space_2_params = options.space2_params;
    const space1_axis_label = options.space1_axis_label;
    const space2_axis_label = options.space2_axis_label;

    const spacing = ((typeof options.spacing === "number") ? options.spacing : 0) + space_1_params.padding;
    const horizontal = (typeof options.horizontal === "boolean") ? options.horizontal : false;

    const space_1_layout = GetLayoutInfoOfSpace(space_1_params);
    const space_2_layout = GetLayoutInfoOfSpace(space_2_params);

    obj.space1 = viz.CreateIterationSpace(space_1_params);
    obj.root_object.attach(obj.space1.root_object);

    if (space1_axis_label) {
        obj.space1_label = viz.CreateAxisLabel({ ...{ space: obj.space1 }, ...space1_axis_label })
        obj.root_object.attach(obj.space1_label.root_object);
    }

    const space2_position = new THREE.Vector3();
    space2_position.z = (space_1_layout.center.z + (space_1_layout.total_dim_size.z / 2)) - (space_2_layout.total_dim_size.z / 2);
    if (horizontal) {
        space2_position.y = (space_1_layout.center.y + (space_1_layout.total_dim_size.y / 2)) - (space_2_layout.total_dim_size.y / 2);
        space2_position.x = (space_1_layout.center.x + (space_1_layout.total_dim_size.x / 2)) + spacing + (space_2_layout.total_dim_size.x / 2);
    } else {
        space2_position.y = (space_1_layout.center.y - (space_1_layout.total_dim_size.y / 2)) - spacing - (space_2_layout.total_dim_size.y / 2);
        space2_position.x = (space_1_layout.center.x - (space_1_layout.total_dim_size.x / 2)) + (space_2_layout.total_dim_size.x / 2);
    }

    space_2_params.position = space2_position;
    obj.space2 = viz.CreateIterationSpace(space_2_params);
    obj.root_object.attach(obj.space2.root_object);

    if (space2_axis_label) {
        obj.space2_label = viz.CreateAxisLabel({ ...{ space: obj.space2 }, ...space2_axis_label })
        obj.root_object.attach(obj.space2_label.root_object);
    }

    return obj;
}

function GetOverlappedPosition(overlapped_space, overlapped_coord, overlapping_space) {
    overlapped_info = GetLayoutInfoOfSpace(overlapped_space);
    overlapping_info = GetLayoutInfoOfSpace(overlapping_space);

    const coord_offset = {
        x: (overlapped_coord[0] * overlapped_space.size) + (overlapped_coord[0] * overlapped_space.padding),
        y: (overlapped_coord[1] * overlapped_space.size) + (overlapped_coord[1] * overlapped_space.padding),
        z: (overlapped_coord[2] * overlapped_space.size) + (overlapped_coord[2] * overlapped_space.padding)
    };

    return {
        x: overlapped_info.origin.y + coord_offset.y + (overlapping_info.total_dim_size.y / 2),
        y: -(overlapped_info.origin.x + coord_offset.x + (overlapping_info.total_dim_size.x / 2)),
        z: -(overlapped_info.origin.z + coord_offset.z + (overlapping_info.total_dim_size.z / 2)),
    };
}


async function AutoSaveFrame(img_name, vizInfo) {
    const frameAsDataURL = document.getElementById("render_container").children[0].toDataURL("image/png");
    if (vizInfo.save_through_browser) {
        img_name = img_name ? img_name : "image";
        console.log("Saving image " + img_name + " through browser");
        document.getElementById("downloader_hidden").download = img_name + ".png";
        document.getElementById("downloader_hidden").href = frameAsDataURL;
        document.getElementById("downloader_hidden").click();
    }

    if (!vizInfo.enable_image) {
        console.log("not saving image on dry run");
        return;
    }

    return PerformRequest(HTTP_METHODS.POST,
        "/api/save_image",
        { frame: frameAsDataURL, viz_name: vizInfo.name, frame_name: img_name })
        .then((result) => {
            console.log("save_image: " + result);
        });
}

async function CreateSequence(viz_obj, sequence_name, fps) {
    const seq_obj = {
        viz_obj: viz_obj,
        viz_name: viz_obj.vizInfo.name,
        sequence_name: sequence_name,
        sequence_fps: fps,
        cur_frame: 0,
        has_finished: false,
        active_tweens: [],
        AddAnimation: (tween) => {
            if (seq_obj.has_finished) {
                console.log("Cannot AddAnimation for finished sequence");
                return seq_obj;
            }

            tween.onComplete(() => {
                const index = seq_obj.active_tweens.indexOf(tween);
                if (index > -1) {
                    seq_obj.active_tweens.splice(index, 1);
                }
            });

            tween.start(0);
            seq_obj.active_tweens.push(tween);
        },
        RenderAllAnimations: async () => {
            if (seq_obj.has_finished) {
                console.log("Cannot RenderAllAnimations for finished sequence");
                return seq_obj;
            }

            let cur_frame = 0;
            const frame_duration = Math.floor(1000 / seq_obj.sequence_fps);
            while (seq_obj.active_tweens.length > 0) {
                TWEEN.update(cur_frame * frame_duration);
                await seq_obj.AddFrame(frame_duration);
                cur_frame++;
            }
        },
        AddFrame: async (duration, save_frame_name) => {
            if (seq_obj.has_finished) {
                console.log("Cannot AddFrame to finished sequence");
                return seq_obj;
            }

            seq_obj.viz_obj.render();

            const save_frame = !!save_frame_name;
            if (save_frame) {
                await seq_obj.viz_obj.SaveImage(save_frame_name);
            }

            if (!seq_obj.viz_obj.vizInfo.enable_video) {
                console.log("not AddFrame on dry run");
                return seq_obj;
            }

            const frameAsDataURL = document.getElementById("render_container").children[0].toDataURL("image/png");
            let frame_info = {
                viz_name: seq_obj.viz_name,
                sequence_name: seq_obj.sequence_name,
                sequence_fps: seq_obj.sequence_fps,
                monotonic_frame: seq_obj.cur_frame++,
                frame: frameAsDataURL
            };

            if (typeof duration === "number") {
                frame_info.frame_duration = duration;
            }

            return PerformRequest(HTTP_METHODS.POST, "/api/add_to_sequence", frame_info)
                .then((result) => {
                    console.log("add_to_sequence result: " + result);
                    return seq_obj;
                });
        },
        FinishSequence: () => {
            if (seq_obj.has_finished) {
                console.log("Cannot FinishSequence on already finished sequence");
                return seq_obj;
            }

            if (!seq_obj.viz_obj.vizInfo.enable_video) {
                console.log("not AddFrame on dry run");
                return seq_obj;
            }

            return PerformRequest(HTTP_METHODS.POST,
                "/api/finish_sequence",
                {
                    viz_name: seq_obj.viz_name,
                    sequence_name: seq_obj.sequence_name,
                    sequence_fps: seq_obj.sequence_fps,
                })
                .then((result) => {
                    console.log("finish_sequence result: " + result);
                    seq_obj.has_finished = true;
                    return seq_obj;
                });
        }
    };

    if (!seq_obj.viz_obj.vizInfo.enable_video) {
        console.log("not AddFrame on dry run");
        return seq_obj;
    }

    return PerformRequest(HTTP_METHODS.POST,
        "/api/start_sequence",
        { viz_name: seq_obj.viz_name, sequence_name: sequence_name, sequence_fps: fps })
        .then((result) => {
            console.log("start_sequence result: " + result);
            return seq_obj;
        });
}

function _create_padding_array(padding) {
    let padding_arr = [];
    if (typeof padding === "number") {
        padding_arr = padding_arr.concat(padding, padding, padding)
    } else if (Array.isArray(padding)) {
        padding_arr = padding_arr.concat(padding);
        if (padding_arr.length < 3) {
            for (let i = padding_arr.length; i < 3; ++i) {
                padding_arr.push(0.2);
            }
        }
    } else {
        padding_arr.fill(0.2, -3)
    }

    return padding_arr;
}

// Helper function for composing multiple iteration spaces in a grid
function Generate4dIterationSpace(viz, options) {
    const inner_params = options.inner_space_params;
    const inner_axis_labels = options.inner_space_axis_labels;
    const inner_space_label_coords = options.inner_space_label_coords ? options.inner_space_label_coords : [];
    const horizontal_spacing = options.horizontal_spacing ? options.horizontal_spacing : 0;
    const vertical_spacing = options.vertical_spacing ? options.vertical_spacing : 0;
    const position = options.position ? options.position : new THREE.Vector3();
    const rotation = options.rotation ? options.rotation : new THREE.Vector3();

    const outer_axis0_size = options.outer_axis0_size ? options.outer_axis0_size : 1;
    const outer_axis0_label = options.outer_axis0;

    const outer_axis1_size = options.outer_axis1_size ? options.outer_axis1_size : 1;
    const outer_axis1_label = options.outer_axis1;

    const inner_space_layout = GetLayoutInfoOfSpace(inner_params);
    const inner_space_padding = _create_padding_array(inner_params.padding);

    let obj = GetBaseSceneObject(viz, options.parent);
    obj.iteration_spaces = [];
    obj.iteration_space_labels = new Array(outer_axis0_size * outer_axis1_size);
    obj.iteration_space_labels.fill(null, 0);
    obj.get_iteration_space = (child_coord) => {
        const dim0_coord = (typeof child_coord[0] === "number") ? child_coord[0] : 0;
        const dim1_coord = (typeof child_coord[1] === "number") ? child_coord[1] : 0;
        return obj.iteration_spaces[(outer_axis1_size * dim0_coord) + dim1_coord];
    };
    obj.get_iteration_space_label = (child_coord) => {
        const dim0_coord = (typeof child_coord[0] === "number") ? child_coord[0] : 0;
        const dim1_coord = (typeof child_coord[1] === "number") ? child_coord[1] : 0;
        return obj.iteration_space_labels[(outer_axis1_size * dim0_coord) + dim1_coord];
    };
    obj.root_object.position.set(
        position.x,
        position.y,
        position.z
    );

    // Ensure all label coords are 2d
    for (let i = 0; i < inner_space_label_coords.length; ++i) {
        if (inner_space_label_coords[i].length > 0) {
            if (inner_space_label_coords[i].length == 1) {
                inner_space_label_coords[i].push(0);
            }
        }
    }

    const overall_width = (outer_axis1_size * inner_space_layout.total_dim_size.x) + ((outer_axis1_size - 1) * (inner_space_padding[1])) + ((outer_axis1_size - 1) * (horizontal_spacing));
    const overall_height = (outer_axis0_size * inner_space_layout.total_dim_size.y) + ((outer_axis0_size - 1) * (inner_space_padding[0])) + ((outer_axis0_size - 1) * (vertical_spacing));

    const getDistanceForCoord = (coord, axis_size, dim_size, spacing, padding) => {
        return (dim_size / 2) + (coord * dim_size) + (coord * spacing) + (coord * padding);
    };

    const starting_position = new THREE.Vector3(position.x - (overall_width / 2), position.y + (overall_height / 2), position.z);
    const ending_position = new THREE.Vector3(position.x + (overall_width / 2), position.y - (overall_height / 2), position.z);

    for (let y = 0; y < outer_axis0_size; ++y) {
        for (let x = 0; x < outer_axis1_size; ++x) {
            let this_space_pos = starting_position.clone();
            this_space_pos.x += getDistanceForCoord(x, 0, inner_space_layout.total_dim_size.x, horizontal_spacing, inner_space_padding[1]);
            this_space_pos.y -= getDistanceForCoord(y, 0, inner_space_layout.total_dim_size.y, vertical_spacing, inner_space_padding[0]);
            inner_params.position = this_space_pos;

            let space = viz.CreateIterationSpace(inner_params);
            obj.iteration_spaces.push(space);
            obj.root_object.attach(space.root_object);

            // Check if we are in a labeled coord
            if (inner_space_label_coords.length > 0) {
                let shouldLabel = false;
                for (let i = 0; i < inner_space_label_coords.length; ++i) {
                    if (inner_space_label_coords[i].length > 0) {
                        shouldLabel = inner_space_label_coords[i][0] == y && inner_space_label_coords[i][1] == x;
                        if (shouldLabel) {
                            break;
                        }
                    }
                }

                if (!shouldLabel) {
                    continue;
                }
            }

            const fullLabelData = { ...{ space: space }, ...inner_axis_labels };
            const label = viz.CreateAxisLabel(fullLabelData);
            obj.iteration_space_labels[(outer_axis1_size * y) + x] = label;
            obj.root_object.attach(label.root_object);
        }
    }

    obj.set_axis0_params = (axis_options) => {
        if (obj.axis0) {
            obj.axis0.remove();
        }

        const distance_from_edge = (typeof axis_options.arrow_distance_from_edge === "number") ? axis_options.arrow_distance_from_edge : 1;
        const edge_start_x = axis_options.arrow_display_side == CUBE_SIDES.LEFT ? starting_position.x : ending_position.x;
        const edge_normal_x = axis_options.arrow_display_side == CUBE_SIDES.LEFT ? -1 : 1;
        const edge_start_z_offset = axis_options.arrow_alignment_side == CUBE_SIDES.FRONT ? (inner_space_layout.total_dim_size.z / 2) : -(inner_space_layout.total_dim_size.z / 2);
        const axis_label = GenerateSingleAxisLabel(viz, null,
            new THREE.Vector3(edge_start_x, starting_position.y, starting_position.z + edge_start_z_offset),
            new THREE.Vector3(edge_normal_x, 0, 0),
            new THREE.Vector3(0, -1, 0),
            distance_from_edge,
            axis_options);

        obj.axis0 = axis_label;
        obj.root_object.attach(axis_label.root_object);
    };

    obj.set_axis1_params = (axis_options) => {
        if (obj.axis1) {
            obj.axis1.remove();
        }

        const distance_from_edge = (typeof axis_options.arrow_distance_from_edge === "number") ? axis_options.arrow_distance_from_edge : 1;
        const edge_start_y = axis_options.arrow_display_side == CUBE_SIDES.TOP ? starting_position.y : ending_position.y;
        const edge_normal_y = axis_options.arrow_display_side == CUBE_SIDES.TOP ? 1 : -1;
        const edge_start_z_offset = axis_options.arrow_alignment_side == CUBE_SIDES.FRONT ? (inner_space_layout.total_dim_size.z / 2) : -(inner_space_layout.total_dim_size.z / 2);
        const axis_label = GenerateSingleAxisLabel(viz, null,
            new THREE.Vector3(starting_position.x, edge_start_y, starting_position.z + edge_start_z_offset),
            new THREE.Vector3(0, edge_normal_y, 0),
            new THREE.Vector3(1, 0, 0),
            distance_from_edge,
            axis_options);

        obj.axis1 = axis_label;
        obj.root_object.attach(axis_label.root_object);
    };

    obj.set_axis_params = (axis_options) => {
        if (options.outer_axis0) {
            obj.set_axis0_params(options.outer_axis0);
        }

        if (options.outer_axis1) {
            obj.set_axis1_params(options.outer_axis1);
        }
    }
    obj.set_axis_params(options);

    obj.remove_all_axis_labels = () => {
        obj.iteration_space_labels.forEach((label) => {
            if (label) {
                label.remove();
            }
        });

        obj.iteration_space_labels.fill(null, 0);

        if (obj.axis0) {
            obj.axis0.remove();
        }

        if (obj.axis1) {
            obj.axis1.remove();
        }
    }

    obj.root_object.rotation.x = rotation.x;
    obj.root_object.rotation.y = rotation.y;
    obj.root_object.rotation.z = rotation.z;

    return obj;
}

// Helper function for creating a group of spheres representing iteration space
function GenerateIterationSpace(viz, options) {
    const parent = options.parent ? options.parent : viz.scene;
    const sphere_color = ValidateColor(options.color, 0x808080);
    const rotation = options.rotation ? options.rotation : new THREE.Vector3();
    const position = options.position ? options.position : new THREE.Vector3();

    var _root = new THREE.Object3D();
    var obj = {
        parent_object: parent,
        root_object: _root,
        child_spheres: [],
        sphere_cameras: [],
        sphere_camera: null,
        render_width: 0,
        render_height: 0,
        side_total_render_size: [],
        initial_position: position,
        creation_params: options,
        set_child_color: (child_coordinate, color, opacity) => {
            color = ValidateColor(color);
            const dim0 = child_coordinate[0] ? child_coordinate[0] : 0;
            const dim1 = child_coordinate[1] ? child_coordinate[1] : 0;
            const dim2 = child_coordinate[2] ? child_coordinate[2] : 0;
            obj.child_spheres[dim0][dim1][dim2].material.color = color;
            obj.set_child_opacity(child_coordinate, opacity);
        },
        set_child_opacity: (child_coordinate, opacity) => {
            const dim0 = child_coordinate.dim0 ? child_coordinate.dim0 : 0;
            const dim1 = child_coordinate.dim1 ? child_coordinate.dim1 : 0;
            const dim2 = child_coordinate.dim2 ? child_coordinate.dim2 : 0;

            const has_opactiy = typeof opacity === 'number';
            if (has_opactiy && opacity < 1.0) {
                obj.child_spheres[dim0][dim1][dim2].material.transparent = true;
                obj.child_spheres[dim0][dim1][dim2].material.opacity = opacity;
            }
            else {
                obj.child_spheres[dim0][dim1][dim2].material.transparent = false;
            }
        },
        get_child_position: (child_coordinate) => {
            let pos = new THREE.Vector3();
            let dim0 = child_coordinate.dim0 ? child_coordinate.dim0 : 0;
            let dim1 = child_coordinate.dim1 ? child_coordinate.dim1 : 0;
            let dim2 = child_coordinate.dim2 ? child_coordinate.dim2 : 0;
            if (Array.isArray(child_coordinate)) {
                dim0 = child_coordinate[0];
                dim1 = child_coordinate[1];
                dim2 = child_coordinate[2];
            }

            obj.child_spheres[dim0][dim1][dim2].getWorldPosition(pos);
            return pos;
        },
        add_arrow: (options) => {
            const positions = options.position;
            let arrow_options = options.arrow_options;
            let tail_child_pos = obj.get_child_position(positions.tail.child_coordinate);
            let head_child_pos = obj.get_child_position(positions.head.child_coordinate);
            const additional_head_offset = typeof arrow_options.head_length === "number" ? arrow_options.head_length / 2 : 0;

            let tail_offset = typeof positions.tail.offset === "number" ? positions.tail.offset : 0;
            tail_offset += (obj.size / 2);

            let head_offset = typeof positions.head.offset === "number" ? positions.head.offset : 0;
            head_offset += (obj.size / 2) + additional_head_offset;

            let full_arrow_distance = tail_child_pos.distanceTo(head_child_pos) - tail_offset - head_offset;
            const arrow_direction = head_child_pos.clone().sub(tail_child_pos).normalize();
            const new_tail = tail_child_pos.clone().add(arrow_direction.clone().multiplyScalar(tail_offset));
            const new_head = new_tail.clone().add(arrow_direction.clone().multiplyScalar(full_arrow_distance));
            arrow_options.tail_position = new_tail;
            arrow_options.head_position = new_head;

            const axis_arrow = GenerateArrow(viz, arrow_options);
            obj.root_object.attach(axis_arrow.root_object);
            return axis_arrow;
        },
        add_multi_arrow: (options) => {
            const positions = options.positions;
            let arrow_options = options.arrow_options;
            const base_head_offset = ValidateNumber(arrow_options.head_offset, 0);
            const additional_head_offset = typeof arrow_options.head_length === "number" ? arrow_options.head_length : 0;
            let all_arrows = [];
            positions.tails.forEach((tail_info, idx) => {
                arrow_options.tail_position = obj.get_child_position(tail_info.child_coordinate);
                arrow_options.head_position = obj.get_child_position(positions.head.child_coordinate);
                arrow_options.head_offset = base_head_offset + additional_head_offset;
                const arrow = GenerateBentArrow(viz, arrow_options, idx != 0);
                obj.root_object.attach(arrow.root_object);
                all_arrows.push(arrow);
            });

            return all_arrows;
        },
        iterate: (fn) => {
            for (let x = 0; x < obj.shape[0]; ++x) {
                for (let y = 0; y < obj.shape[1]; ++y) {
                    for (let z = 0; z < obj.shape[2]; ++z) {
                        fn(x, y, z, obj.child_spheres[x][y][z], [x, y, z]);
                    }
                }
            }
        },
        remove: () => { obj.parent_object.remove(obj.root_object); }
    };

    obj.create_space = (space_params) => {
        obj.shape = space_params.shape ? space_params.shape : [1, 1, 1];
        obj.size = space_params.size ? space_params.size : 1;

        obj.padding = _create_padding_array(space_params.padding);
        console.log(obj.padding)

        // Handle 1d / 2d
        let not3d = obj.shape.length < 3;
        if (not3d) {
            for (let i = obj.shape.length; i < 3; ++i) {
                obj.shape.push(1);
            }
        }

        const overall_x = (obj.shape[0]) * obj.size + (obj.shape[0] * obj.padding[1]);
        const overall_y = (obj.shape[1]) * obj.size + (obj.shape[1] * obj.padding[0]);
        const overall_z = (obj.shape[2]) * obj.size + (obj.shape[2] * obj.padding[2]);

        const space_info = GetLayoutInfoOfSpace(space_params);

        obj.side_total_render_size = [space_info.total_dim_size.x, space_info.total_dim_size.y, space_info.total_dim_size.z];
        if (obj.child_spheres.length > 0) {
            for (let x = 0; x < obj.shape[0]; ++x) {
                for (let y = 0; y < obj.shape[1]; ++y) {
                    for (let z = 0; z < obj.shape[2]; ++z) {
                        obj.root_object.remove(obj.child_spheres[x][y][z]);
                    }
                }
            }
        }

        // Handle 3d
        if (obj.shape.length == 3) {
            const starting_pos = [space_info.total_dim_size.x / 2, space_info.total_dim_size.y / 2, space_info.total_dim_size.z / 2];

            console.log("Creating 3d iteration space");

            obj.child_spheres = new Array(obj.shape[0]);
            for (let x = 0; x < obj.shape[0]; ++x) {
                obj.child_spheres[x] = new Array(obj.shape[1]);
                for (let y = 0; y < obj.shape[1]; ++y) {
                    obj.child_spheres[x][y] = new Array(obj.shape[2]);
                }
            }

            for (let x = 0; x < obj.shape[0]; ++x) {
                for (let y = 0; y < obj.shape[1]; ++y) {
                    for (let z = 0; z < obj.shape[2]; ++z) {
                        const spherePos = new THREE.Vector3(
                            -(starting_pos[0]) + (obj.size / 2) + (obj.padding[1] / 2) + (y * obj.size) + (y * obj.padding[1]),
                            -(starting_pos[1]) + (obj.size / 2) + (obj.padding[0] / 2) + (x * obj.size) + (x * obj.padding[0]),
                            -(starting_pos[2]) + (obj.size / 2) + (obj.padding[2] / 2) + (z * obj.size) + (z * obj.padding[2]),
                        );

                        const geometry = new THREE.SphereGeometry(obj.size / 2, 32, 32);
                        const material = new THREE.MeshStandardMaterial({ color: sphere_color });
                        const sphere = new THREE.Mesh(geometry, material);
                        sphere.position.x = spherePos.x;
                        sphere.position.y = spherePos.y;
                        sphere.position.z = spherePos.z;
                        sphere.castShadow = true;
                        sphere.receiveShadow = true;

                        obj.root_object.add(sphere);
                        obj.child_spheres[x][y][z] = sphere;
                    }
                }
            }

            if (!not3d) {
                obj.render_width = Math.sqrt((overall_x * overall_x) + (overall_y * overall_y));
                obj.render_height = Math.sqrt((overall_x * overall_x) + (overall_y * overall_y));
            }
            else {
                obj.render_width = overall_x;
                obj.render_height = overall_y;
            }
        }
    }

    obj.set_padding = (padding) => {
        obj.creation_params.padding = _create_padding_array(space_params.padding);
        obj.create_space(obj.creation_params);
    }

    obj.create_space(options);

    obj.root_object.rotation.x = rotation.x;
    obj.root_object.rotation.y = rotation.y;
    obj.root_object.rotation.z = rotation.z;

    obj.parent_object.add(obj.root_object);

    obj.root_object.position.x = position.x;
    obj.root_object.position.y = position.y;
    obj.root_object.position.z = position.z;


    obj.axis0_axis1_face_normal = () => { return new THREE.Vector3(0, 0, -1).applyQuaternion(obj.root_object.quaternion); }
    //GenerateArrow(viz.scene, obj.root_object.position.clone().add(obj.axis0_axis1_face_normal.clone().multiplyScalar(50)), obj.root_object.position.clone(), 0.25, 0.5, 0.5, 0xFF0000, 0xFF0000, true);

    obj.axis0_axis2_face_normal = () => { return new THREE.Vector3(0, -1, 0).applyQuaternion(obj.root_object.quaternion); }
    //GenerateArrow(viz.scene, obj.root_object.position.clone().add(obj.axis0_axis2_face_normal.clone().multiplyScalar(50)), obj.root_object.position.clone(), 0.25, 0.5, 0.5, 0x00FF00, 0x00FF00, true);

    obj.axis1_axis2_face_normal = () => { return new THREE.Vector3(1, 0, 0).applyQuaternion(obj.root_object.quaternion); }
    //GenerateArrow(viz.scene, obj.root_object.position.clone().add(obj.axis1_axis2_face_normal.clone().multiplyScalar(50)), obj.root_object.position.clone(), 0.25, 0.5, 0.5, 0x0000FF, 0x0000FF, true);

    return obj;
}

const GenerateSingleAxisLabel = (viz, parent, edge_start_pos, edge_normal, edge_direction, edge_padding, axis_info) => {
    const arrow_distance_from_edge = (typeof axis_info.arrow_distance_from_edge === "number") ? axis_info.arrow_distance_from_edge : 1;
    const arrow_start_offset = (typeof axis_info.arrow_start_offset === "number") ? axis_info.arrow_start_offset : 0;

    let head_pos, tail_pos, text_pos;
    tail_pos = edge_start_pos.clone().add(edge_normal.clone().multiplyScalar(arrow_distance_from_edge + axis_info.arrow_thickness))
    tail_pos = tail_pos.clone().add(edge_direction.clone().multiplyScalar(arrow_start_offset))
    head_pos = tail_pos.clone().add(edge_direction.clone().multiplyScalar(axis_info.arrow_length));
    text_pos = tail_pos.clone().add(head_pos.clone().sub(tail_pos).normalize().multiplyScalar((axis_info.label_pos * axis_info.arrow_length)));
    text_pos = text_pos.clone().add(edge_normal.clone().multiplyScalar(axis_info.label_size));

    const arrowhead_length = axis_info.arrowhead_length ? axis_info.arrowhead_length : axis_info.arrowhead_thickness;

    var obj = GetBaseSceneObject(viz, parent);

    const axis_arrow = GenerateArrow(viz, {
        head_position: head_pos,
        tail_position: tail_pos,
        line_thickness: axis_info.arrow_thickness,
        head_length: arrowhead_length,
        head_thickness: axis_info.arrowhead_thickness,
        line_color: axis_info.color,
        head_color: axis_info.color,
        ignore_lighting: true
    });

    const axis_label = GenerateText(viz, {
        position: text_pos,
        text: axis_info.label,
        size: axis_info.label_size,
        font_family: axis_info.label_font,
        color: axis_info.color
    });

    obj.root_object.attach(axis_arrow.root_object);
    obj.root_object.attach(axis_label.root_object);
    return obj;
};

function SlerpVectors(vec1, vec2, amount) {
    var mx1 = new THREE.Matrix4().lookAt(vec1, new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1, 0));
    var qt1 = new THREE.Quaternion().setFromRotationMatrix(mx1);

    var mx2 = new THREE.Matrix4().lookAt(vec2, new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1, 0));
    var qt2 = new THREE.Quaternion().setFromRotationMatrix(mx2);

    let slerped_normal = new THREE.Vector3(0, 0, 1);
    slerped_normal = slerped_normal.applyQuaternion(qt1.slerp(qt2, amount));
    return slerped_normal;
}

function GenerateAxisLabel(viz, options) {
    console.log("Creating iteration space label");

    const iter_space = options.space;
    const iter_space_layout = GetLayoutInfoOfSpace(iter_space.creation_params);
    const combined_space = (iter_space.size + iter_space.padding);
    const position = options.position ? options.position : new THREE.Vector3();

    var obj = GetBaseSceneObject(viz, options.parent);
    obj.root_object.position.set(
        iter_space.root_object.position.x,
        iter_space.root_object.position.y,
        iter_space.root_object.position.z
    );

    obj.set_axis_params = (axis_options) => {
        if (options.axis0) {
            obj.set_axis0_params(options.axis0);
        }
    }


    obj.set_axis0_params = (axis_options) => {
        if (obj.axis0) {
            obj.axis0.remove();
        }

        const edge_angle = (typeof axis_options.edge_angle === "number") ? axis_options.edge_angle : 0;
        const edge_normal_base = axis_options.arrow_display_side == CUBE_SIDES.LEFT ? iter_space.axis1_axis2_face_normal().clone().multiplyScalar(-1) : iter_space.axis1_axis2_face_normal().clone();
        const edge_normal_full = iter_space.axis0_axis1_face_normal().clone();
        let edge_normal_new = SlerpVectors(edge_normal_base, edge_normal_full, edge_angle);

        const dim1_start_coord = axis_options.arrow_display_side == CUBE_SIDES.LEFT ? 0 : iter_space.shape[1] - 1;
        const dim2_start_coord = axis_options.arrow_alignment_side == CUBE_SIDES.FRONT ? 0 : iter_space.shape[2] - 1;
        const upper_left_sphere = iter_space.get_child_position({ dim0: 0, dim1: dim1_start_coord, dim2: dim2_start_coord });
        const edge_normal = axis_options.arrow_display_side == CUBE_SIDES.LEFT ? iter_space.axis1_axis2_face_normal().clone().multiplyScalar(-1) : iter_space.axis1_axis2_face_normal().clone();
        const axis0 = GenerateSingleAxisLabel(
            viz,
            obj.root_object,
            upper_left_sphere,
            edge_normal_new,
            iter_space.axis0_axis2_face_normal().clone().multiplyScalar(-1),
            combined_space,
            axis_options
        );

        obj.axis0 = axis0;
        obj.root_object.attach(axis0.root_object);
    };

    obj.set_axis1_params = (axis_options) => {
        if (obj.axis1) {
            obj.axis1.remove();
        }

        const edge_angle = (typeof axis_options.edge_angle === "number") ? axis_options.edge_angle : 0;
        const edge_normal_base = axis_options.arrow_display_side == CUBE_SIDES.TOP ? iter_space.axis0_axis2_face_normal().clone() : iter_space.axis0_axis2_face_normal().clone().multiplyScalar(-1);
        const edge_normal_full = iter_space.axis0_axis1_face_normal().clone();
        let edge_normal_new = SlerpVectors(edge_normal_base, edge_normal_full, edge_angle);

        const dim0_start_coord = axis_options.arrow_display_side == CUBE_SIDES.TOP ? 0 : iter_space.shape[0] - 1;
        const dim2_start_coord = axis_options.arrow_alignment_side == CUBE_SIDES.FRONT ? 0 : iter_space.shape[2] - 1;
        const edge_normal = axis_options.arrow_display_side == CUBE_SIDES.TOP ? iter_space.axis0_axis2_face_normal().clone() : iter_space.axis0_axis2_face_normal().clone().multiplyScalar(-1);
        const lower_left_sphere = iter_space.get_child_position({ dim0: dim0_start_coord, dim1: 0, dim2: dim2_start_coord });
        const axis1 = GenerateSingleAxisLabel(
            viz,
            obj.root_object,
            lower_left_sphere,
            edge_normal_new,
            iter_space.axis1_axis2_face_normal().clone(),
            combined_space,
            axis_options);

        obj.axis1 = axis1;
        obj.root_object.attach(axis1.root_object);
    };

    obj.set_axis2_params = (axis_options) => {
        if (obj.axis2) {
            obj.axis2.remove();
        }

        const edge_angle = (typeof axis_options.edge_angle === "number") ? axis_options.edge_angle : 0;
        const edge_normal_base = (axis_options.arrow_display_side == CUBE_SIDES.LEFT) ? iter_space.axis1_axis2_face_normal().clone().multiplyScalar(-1) : iter_space.axis1_axis2_face_normal().clone();
        const edge_normal_full = iter_space.axis0_axis2_face_normal().clone().multiplyScalar(-1);
        let edge_normal_new = SlerpVectors(edge_normal_base, edge_normal_full, edge_angle);

        const dim1_start_coord = axis_options.arrow_display_side == CUBE_SIDES.LEFT ? 0 : iter_space.shape[1] - 1;
        const lower_center_sphere = iter_space.get_child_position({ dim0: iter_space.shape[0] - 1, dim1: dim1_start_coord, dim2: 0 });
        const axis2 = GenerateSingleAxisLabel(
            viz,
            obj.root_object,
            lower_center_sphere,
            edge_normal_new,
            iter_space.axis0_axis1_face_normal().clone().multiplyScalar(-1),
            combined_space,
            axis_options);

        obj.axis2 = axis2;
        obj.root_object.attach(axis2.root_object);
    };

    obj.set_axis_params = (axis_options) => {
        if (options.axis0) {
            obj.set_axis0_params(options.axis0);
        }

        if (options.axis1) {
            obj.set_axis1_params(options.axis1);
        }

        if (options.axis2) {
            obj.set_axis2_params(options.axis2);
        }
    }

    obj.set_axis_params(options);

    iter_space.root_object.attach(obj.root_object);
    return obj;
}

function GenerateIterationSpaceLighting(viz, options) {
    const show_lights = options.show_lights ? true : false;
    const iter_space = options.space;

    const shadow_map_resolution = (typeof viz.vizInfo.shadow_map_resolution === "number") ? viz.vizInfo.shadow_map_resolution : 512;

    var obj = GetBaseSceneObject(viz, options.parent);
    obj.root_object.position.set(iter_space.root_object.position.x, iter_space.root_object.position.y, iter_space.root_object.position.z);

    // {color: 0x000000, panel_distance: ,intensity : Float, distance : Number, decay : Float  light_count: 5, light_stride: 5, offset: {x: 0, y: 0}}
    const CreateLightPanel = (side_center, side_normal, side_up, side_right, side_height, side_width, light_options) => {
        const color = light_options.color ? light_options.color : 0xFFFFFF;
        const intensity = light_options.intensity ? light_options.intensity : 1;
        const distance = light_options.distance ? light_options.distance : 0;
        const decay = light_options.decay ? light_options.decay : 1;
        const panel_distance = light_options.panel_distance ? light_options.panel_distance : 0;

        let half_grid_size_x = (light_options.light_count.x - 1) * light_options.light_stride.x / 2;
        let half_grid_size_y = (light_options.light_count.y - 1) * light_options.light_stride.y / 2;

        let starting_corner = side_center.clone().add(side_normal.clone().multiplyScalar(panel_distance));
        starting_corner = starting_corner.add(side_right.clone().multiplyScalar(-half_grid_size_x)).add(side_up.clone().multiplyScalar(-half_grid_size_y));
        starting_corner = starting_corner.add(new THREE.Vector3(light_options.offset.x, light_options.offset.y))
        for (var x = 0; x < light_options.light_count.x; ++x) {
            for (var y = 0; y < light_options.light_count.y; ++y) {
                const this_light_pos = new THREE.Vector3(x * light_options.light_stride.x, y * light_options.light_stride.y);
                const light = new THREE.PointLight(color, intensity, distance, decay);
                const final_pos = starting_corner.clone().add(side_right.clone().multiplyScalar(this_light_pos.x)).add(side_up.clone().multiplyScalar(this_light_pos.y));;
                light.position.set(final_pos.x, final_pos.y, final_pos.z);
                light.castShadow = true;
                light.shadow.mapSize.width = light.shadow.mapSize.height = shadow_map_resolution;
                obj.root_object.add(light);

                if (show_lights) {
                    const geometry = new THREE.SphereGeometry(0.5, 32, 32);
                    const material = new THREE.MeshStandardMaterial({ color: 0xFF8C00, emissive: 0xFF8C00, emissiveIntensity: 5 });
                    const sphere = new THREE.Mesh(geometry, material);
                    sphere.position.set(final_pos.x, final_pos.y, final_pos.z);
                    sphere.castShadow = false;
                    sphere.receiveShadow = false;
                    obj.root_object.add(sphere);
                }

            }
        }
    };

    const axis0_side = iter_space.side_total_render_size[0];
    const axis1_side = iter_space.side_total_render_size[1];
    const axis2_side = iter_space.side_total_render_size[2];
    let side_normal, side_up, side_right, center_point;

    // Create top light panel
    side_normal = iter_space.axis0_axis2_face_normal().clone();
    side_up = iter_space.axis1_axis2_face_normal().clone().multiplyScalar(-1);
    side_right = iter_space.axis0_axis1_face_normal().clone().multiplyScalar(-1);
    center_point = iter_space.root_object.position.clone().add(side_normal.clone().multiplyScalar(axis0_side / 2));

    CreateLightPanel(center_point, side_normal, side_up, side_right, axis1_side, axis2_side, options.top_light)

    // Create left light panel
    side_normal = iter_space.axis0_axis1_face_normal().clone();
    side_up = iter_space.axis0_axis2_face_normal().clone();
    side_right = iter_space.axis1_axis2_face_normal().clone();
    center_point = iter_space.root_object.position.clone().add(side_normal.clone().multiplyScalar(axis2_side / 2));

    CreateLightPanel(center_point, side_normal, side_up, side_right, axis1_side, axis0_side, options.left_light)

    // Create right light panel
    side_normal = iter_space.axis1_axis2_face_normal().clone();
    side_up = iter_space.axis0_axis2_face_normal().clone();
    side_right = iter_space.axis0_axis1_face_normal().clone().multiplyScalar(1);
    center_point = iter_space.root_object.position.clone().add(side_normal.clone().multiplyScalar(axis1_side / 2));

    CreateLightPanel(center_point, side_normal, side_up, side_right, axis1_side, axis2_side, options.right_light)

    return obj;
}

function GetSideInfo(cube_side) {
    let result = {};
    if (cube_side === CUBE_SIDES.FRONT || cube_side === CUBE_SIDES.BACK) {
        const z_norm = (cube_side === CUBE_SIDES.FRONT) ? 1 : -1;
        result.side_up = new THREE.Vector3(0, 1, 0);
        result.side_right = new THREE.Vector3(1, 0, 0);
        result.side_normal = new THREE.Vector3(0, 0, z_norm);
    } else if (cube_side === CUBE_SIDES.LEFT || cube_side === CUBE_SIDES.RIGHT) {
        const x_norm = (cube_side === CUBE_SIDES.LEFT) ? -1 : 1;
        result.side_up = new THREE.Vector3(0, 1, 0);
        result.side_right = new THREE.Vector3(0, 0, -1);
        result.side_normal = new THREE.Vector3(x_norm, 0, 0);
    } else if (cube_side === CUBE_SIDES.TOP || cube_side === CUBE_SIDES.BOTTOM) {
        const y_norm = (cube_side === CUBE_SIDES.TOP) ? 1 : -1;
        result.side_up = new THREE.Vector3(0, 0, -1);
        result.side_right = new THREE.Vector3(1, 0, 0);
        result.side_normal = new THREE.Vector3(0, y_norm, 0);
    }

    return result;
}

// options: {position, rotation, light_panels: []}
// top_light: {color: 0xCCCCFF, panel_distance: 16, intensity : 1.35, distance : 28, decay : 2,  light_count: {x: 2, y: 2}, light_stride: {x: 12, y: 16}, offset: {x: -1, y: 0}},
function GenerateLightingCube(viz, options) {
    const position = options.position ? options.position : new THREE.Vector3();
    const rotation = options.rotation ? options.rotation : new THREE.Vector3();
    const shadow_map_resolution = (typeof viz.vizInfo.shadow_map_resolution === "number") ? viz.vizInfo.shadow_map_resolution : 512;

    var obj = GetBaseSceneObject(viz, options.parent);

    const GenerateLightingPanel = (panel_options) => {
        const show_lights = panel_options.show_lights ? true : false;
        const light_type = (panel_options.light_type === LIGHT_TYPES.DIRECTIONAL) ? LIGHT_TYPES.DIRECTIONAL : LIGHT_TYPES.POINT;

        const color = ValidateColor(panel_options.color, 0xFFFFFF);
        const intensity = (typeof panel_options.intensity === "number") ? panel_options.intensity : 1;
        const distance = 0;//(typeof light_options.distance === "number") ? light_options.distance : 0;
        const decay = (typeof panel_options.decay === "number") ? panel_options.decay : 1;
        const panel_distance = (typeof panel_options.panel_distance === "number") ? panel_options.panel_distance : 0;

        let half_grid_size_x = (panel_options.light_count.x - 1) * panel_options.light_stride.x / 2;
        let half_grid_size_y = (panel_options.light_count.y - 1) * panel_options.light_stride.y / 2;

        const side_info = GetSideInfo(panel_options.cube_side);

        let panel_center = side_info.side_normal.clone().multiplyScalar(panel_distance);
        panel_center = panel_center.clone().add(side_info.side_up.clone().multiplyScalar(panel_options.offset.y))
        panel_center = panel_center.clone().add(side_info.side_right.clone().multiplyScalar(panel_options.offset.x))

        let starting_corner = panel_center.clone().add(side_info.side_up.clone().multiplyScalar(-half_grid_size_y));
        starting_corner = starting_corner.clone().add(side_info.side_right.clone().multiplyScalar(-half_grid_size_x));

        for (var x = 0; x < panel_options.light_count.x; ++x) {
            for (var y = 0; y < panel_options.light_count.y; ++y) {
                const this_light_pos = new THREE.Vector3(x * panel_options.light_stride.x, y * panel_options.light_stride.y);


                let light = null;
                if (light_type === LIGHT_TYPES.POINT) {
                    light = new THREE.PointLight(color, intensity, distance, decay);
                } else {
                    light = new THREE.DirectionalLight(color, intensity);
                }

                const final_pos = starting_corner.clone().add(side_info.side_right.clone().multiplyScalar(this_light_pos.x)).add(side_info.side_up.clone().multiplyScalar(this_light_pos.y));
                light.position.set(final_pos.x, final_pos.y, final_pos.z);
                light.castShadow = true;
                light.shadow.mapSize.width = light.shadow.mapSize.height = shadow_map_resolution;
                obj.root_object.add(light);

                if (light_type === LIGHT_TYPES.DIRECTIONAL) {
                    const frustrumSize = (typeof panel_options.shadow_area_size === "number") ? panel_options.shadow_area_size : 25;
                    light.shadow.camera.left = -frustrumSize;
                    light.shadow.camera.right = frustrumSize;
                    light.shadow.camera.top = frustrumSize;
                    light.shadow.camera.bottom = -frustrumSize;
                    obj.root_object.add(light.target);
                }

                if (show_lights) {
                    const geometry = new THREE.SphereGeometry(0.5, 32, 32);
                    const material = new THREE.MeshStandardMaterial({ color: color, emissive: color, emissiveIntensity: 5 });
                    const sphere = new THREE.Mesh(geometry, material);
                    sphere.position.set(final_pos.x, final_pos.y, final_pos.z);
                    sphere.castShadow = false;
                    sphere.receiveShadow = false;
                    if (light_type === LIGHT_TYPES.DIRECTIONAL) {
                        viz.scene.add(new THREE.CameraHelper(light.shadow.camera));
                    }
                    obj.root_object.add(sphere);
                }

            }
        }
    };

    if (options.light_panels) {
        options.light_panels.forEach((panel_options) => {
            GenerateLightingPanel(panel_options);
        });
    }

    obj.root_object.position.x = position.x;
    obj.root_object.position.y = position.y;
    obj.root_object.position.z = position.z;

    obj.root_object.rotation.x = rotation.x;
    obj.root_object.rotation.y = rotation.y;
    obj.root_object.rotation.z = rotation.z;

    return obj;
}

function CreateThreeLightFromOptions(viz, light_options, directional_target_parent) {
    const light_type = (light_options.light_type === LIGHT_TYPES.DIRECTIONAL) ? LIGHT_TYPES.DIRECTIONAL : LIGHT_TYPES.POINT;
    const color = ValidateColor(light_options.color, 0xFFFFFF);
    const intensity = (typeof light_options.intensity === "number") ? light_options.intensity : 1;
    const decay = (typeof light_options.decay === "number") ? light_options.decay : 1;
    const frustrumSize = ((typeof light_options.shadow_area_size === "number") ? light_options.shadow_area_size : 50) / 2;
    const show_light = light_options.show_light ? true : false;
    const shadow_map_resolution = (typeof viz.vizInfo.shadow_map_resolution === "number") ? viz.vizInfo.shadow_map_resolution : 512;

    let three_light = null;
    if (light_type === LIGHT_TYPES.POINT) {
        three_light = new THREE.PointLight(color, intensity, 0, decay);
    } else {
        three_light = new THREE.DirectionalLight(color, intensity);
    }

    three_light.castShadow = true;
    three_light.shadow.mapSize.width = three_light.shadow.mapSize.height = shadow_map_resolution;

    if (light_type === LIGHT_TYPES.DIRECTIONAL) {
        three_light.shadow.camera.left = -frustrumSize;
        three_light.shadow.camera.right = frustrumSize;
        three_light.shadow.camera.top = frustrumSize;
        three_light.shadow.camera.bottom = -frustrumSize;
        console.log("Far plane " + three_light.shadow.camera.far)
        directional_target_parent.add(three_light.target);
    }

    if (show_light) {
        const geometry = new THREE.SphereGeometry(0.5, 32, 32);
        const material = new THREE.MeshStandardMaterial({ color: color, emissive: color, emissiveIntensity: 5 });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.castShadow = false;
        sphere.receiveShadow = false;
        three_light.add(sphere);
        if (light_type === LIGHT_TYPES.DIRECTIONAL) {
            viz.scene.add(new THREE.CameraHelper(three_light.shadow.camera));
        }
    }

    return three_light;
}

// options: {position, rotation, vertical_offset, radius, lights: []}
// light: {color: 0xCCCCFF, intensity : 1.35, distance : 28, decay : 2},
function GenerateLightingRing(viz, options) {
    const position = options.position ? options.position : new THREE.Vector3();
    const rotation = options.rotation ? options.rotation : new THREE.Vector3();
    const vertical_offset = (typeof options.vertical_offset === "number") ? options.vertical_offset : 10;
    const radius = (typeof options.radius === "number") ? options.radius : 10;

    var obj = GetBaseSceneObject(viz, options.parent);
    obj.root_object.position.x = position.x;
    obj.root_object.position.y = position.y;
    obj.root_object.position.z = position.z;

    obj.ring_center = new THREE.Object3D();
    obj.radius = radius;
    obj.lights = [];

    obj.set_vertical_offset = (offset) => {
        obj.vertical_offset = offset;
        obj.root_object.remove(obj.ring_center);
        obj.ring_center.position.x = obj.root_object.position.x;
        obj.ring_center.position.y = obj.root_object.position.y + obj.vertical_offset;
        obj.ring_center.position.z = obj.root_object.position.z;
        obj.root_object.attach(obj.ring_center);
    }

    obj.add_light_to_ring = (light_options) => {
        var light = GetBaseSceneObject(viz);
        light.set_angle = (angle) => {
            obj.ring_center.remove(light.root_object);
            const DegToRad = (Math.PI / 180);
            const angleInRad = angle * DegToRad;
            x = obj.ring_center.position.x + (obj.radius * Math.cos(angleInRad))
            z = obj.ring_center.position.z + (obj.radius * Math.sin(angleInRad))

            light.root_object.position.x = x;
            light.root_object.position.y = obj.ring_center.position.y;
            light.root_object.position.z = z;
            obj.ring_center.attach(light.root_object);
        }

        let three_light = CreateThreeLightFromOptions(viz, light_options, obj.root_object);
        light.root_object.add(three_light);
        light.set_angle(light_options.angle);
        obj.lights.push(light);
    }

    options.lights.forEach((light_options) => {
        obj.add_light_to_ring(light_options);
    });

    obj.set_vertical_offset(vertical_offset);
    return obj;
}

function getAxisDirectionV3(axis) {
    if (axis === AXIS._0) {
        return new THREE.Vector3(0, 1, 0);
    } else if (axis === AXIS._1) {
        return new THREE.Vector3(1, 0, 0);
    } else if (axis === AXIS._2) {
        return new THREE.Vector3(0, 0, 1);
    }

    return new THREE.Vector3(0, 0, 0);
}

function getRelevantComponentForAxis(axis, vector) {
    if (axis === AXIS._0) {
        return vector.y;
    } else if (axis === AXIS._1) {
        return vector.x;
    } else if (axis === AXIS._2) {
        return vector.z;
    }

    return 0;
}

function GenerateBentArrow(viz, options, skip_cone) {
    var obj = GetBaseSceneObject(viz, options.parent)

    const head_pos = options.head_position ? options.head_position : new THREE.Vector3();
    const tail_pos = options.tail_position ? options.tail_position : new THREE.Vector3();
    const tail_offset = ValidateNumber(options.tail_offset, 0);
    const tail_bend_distance = ValidateNumber(options.tail_bend_distance, 0);
    const head_offset = ValidateNumber(options.head_offset, 0);
    const head_bend_distance = ValidateNumber(options.head_bend_distance, 0);
    const spline_curve_type = typeof options.spline_curve_type === "string" ? options.spline_curve_type : "centripetal";
    const spline_tension = ValidateNumber(options.spline_tension, 0.25);
    const axis_alignment = options.axis_alignment ? options.axis_alignment : AXIS._0;
    const line_thickness = options.line_thickness ? options.line_thickness : 0.5;
    const head_length = options.head_length ? options.head_length : 1;
    const head_thickness = options.head_thickness ? options.head_thickness : head_length;
    const line_color = ValidateColor(options.line_color);
    const head_color = ValidateColor(options.head_color);
    const ignore_lighting = options.ignore_lighting ? options.ignore_lighting : false;

    let HeadPosV3 = new THREE.Vector3(head_pos.x, head_pos.y, head_pos.z);
    let TailPosV3 = new THREE.Vector3(tail_pos.x, tail_pos.y, tail_pos.z);
    const TailToHead = TailPosV3.clone().sub(HeadPosV3).normalize();

    const TailBendDir = getAxisDirectionV3(axis_alignment).multiplyScalar(-Math.sign(getRelevantComponentForAxis(axis_alignment, TailToHead)));
    const HeadBendDir = TailBendDir.clone().multiplyScalar(-1);

    HeadPosV3 = HeadPosV3.add(HeadBendDir.clone().multiplyScalar(head_offset));
    TailPosV3 = TailPosV3.add(TailBendDir.clone().multiplyScalar(tail_offset));

    if (!skip_cone) {
        const cone_geo = new THREE.ConeGeometry(head_thickness / 2, head_length, 32);
        let cone_mat = new THREE.MeshPhongMaterial({ color: head_color, specular: 0x050505, shininess: 150 });
        if (ignore_lighting) { cone_mat = new THREE.MeshBasicMaterial({ color: head_color }); }
        const cone = new THREE.Mesh(cone_geo, cone_mat);
        cone.position.x = HeadPosV3.x;
        cone.position.y = HeadPosV3.y;
        cone.position.z = HeadPosV3.z;
        var axis = new THREE.Vector3(0, 1, 0);
        cone.quaternion.setFromUnitVectors(axis, TailBendDir.clone().normalize());
        obj.root_object.add(cone);
    }

    const curve = new THREE.CatmullRomCurve3([
        TailPosV3.clone(),
        TailPosV3.clone().add(TailBendDir.clone().multiplyScalar(tail_bend_distance)),
        HeadPosV3.clone().add(HeadBendDir.clone().multiplyScalar(head_bend_distance)),
        HeadPosV3.clone(),
    ], false, spline_curve_type, spline_tension);

    // path: path 40: number of subdivisions along the track 2: pipe radius 25: number of subdivisions of the pipe section circle
    var tube_geo = new THREE.TubeGeometry(curve, 40, line_thickness / 2, 25, false);
    let tube_mat = new THREE.MeshPhongMaterial({ color: line_color, specular: 0x050505, shininess: 150 });
    if (ignore_lighting) { tube_mat = new THREE.MeshBasicMaterial({ color: line_color }); }
    const tube_mesh = new THREE.Mesh(tube_geo, tube_mat);

    obj.root_object.add(tube_mesh);

    obj.parent_object.add(obj.root_object);

    return obj;
}

// Helper function for creating an arrow pointing at a position
function GenerateArrow(viz, options) {
    const parent = options.parent ? options.parent : viz.scene;
    var _root = new THREE.Object3D();
    var obj = {
        parent_object: parent,
        root_object: _root,
        remove: () => { obj.parent_object.remove(obj.root_object); }
    };

    const head_pos = options.head_position ? options.head_position : new THREE.Vector3();
    const tail_pos = options.tail_position ? options.tail_position : new THREE.Vector3();
    const line_thickness = options.line_thickness ? options.line_thickness : 0.5;
    const head_length = options.head_length ? options.head_length : 1;
    const head_thickness = options.head_thickness ? options.head_thickness : head_length;
    const line_color = ValidateColor(options.line_color);
    const head_color = ValidateColor(options.head_color);
    const ignore_lighting = options.ignore_lighting ? options.ignore_lighting : false;

    const HeadPosV3 = new THREE.Vector3(head_pos.x, head_pos.y, head_pos.z);
    const TailPosV3 = new THREE.Vector3(tail_pos.x, tail_pos.y, tail_pos.z);

    const length = HeadPosV3.distanceTo(TailPosV3);
    const pos = TailPosV3.clone().add(HeadPosV3.clone().sub(TailPosV3).normalize().multiplyScalar(length / 2));

    const cyl_geo = new THREE.CylinderGeometry(line_thickness / 2, line_thickness / 2, length, 32);
    let cyl_mat = new THREE.MeshPhongMaterial({ color: line_color, specular: 0x050505, shininess: 150 });
    if (ignore_lighting) { cyl_mat = new THREE.MeshBasicMaterial({ color: line_color }); }
    const cyl = new THREE.Mesh(cyl_geo, cyl_mat);
    cyl.position.x = pos.x;
    cyl.position.y = pos.y;
    cyl.position.z = pos.z;
    var axis = new THREE.Vector3(0, 1, 0);
    cyl.quaternion.setFromUnitVectors(axis, HeadPosV3.clone().sub(TailPosV3).normalize());
    obj.root_object.add(cyl);

    const cone_geo = new THREE.ConeGeometry(head_thickness / 2, head_length, 32);
    let cone_mat = new THREE.MeshPhongMaterial({ color: head_color, specular: 0x050505, shininess: 150 });
    if (ignore_lighting) { cone_mat = new THREE.MeshBasicMaterial({ color: head_color }); }
    const cone = new THREE.Mesh(cone_geo, cone_mat);
    cone.position.x = HeadPosV3.x;
    cone.position.y = HeadPosV3.y;
    cone.position.z = HeadPosV3.z;
    var axis = new THREE.Vector3(0, 1, 0);
    cone.quaternion.setFromUnitVectors(axis, HeadPosV3.clone().sub(TailPosV3).normalize());
    obj.root_object.add(cone);

    obj.parent_object.add(obj.root_object);

    return obj;
}

function GenerateLine(viz, options) {
    return GenerateArrow(viz, {
        head_position: options.start_position,
        tail_position: options.end_position,
        line_thickness: options.line_thickness,
        head_length: 0.1,
        head_thickness: 0.1,
        line_color: options.line_color,
        head_color: options.line_color,
        ignore_lighting: options.ignore_lighting
    });
}

function ValidateColor(color, default_color) {
    default_color = default_color ? default_color : 0x000000;
    let tjsColor = new THREE.Color(default_color);

    if (!(color === undefined || color === null)) {
        if (typeof color === 'number') {
            tjsColor = new THREE.Color(color);
        } else if (typeof color === 'string') {
            tjsColor = new THREE.Color(color);
        }
    }

    return tjsColor;
}

function ValidateNumber(number, default_value) {

    if (typeof number === "number") {
        return number;
    }

    return default_value;
}

function GenerateText(viz, options) {
    const parent = options.parent ? options.parent : viz.scene;
    var _root = new THREE.Object3D();
    var obj = {
        parent_object: parent,
        root_object: _root,
        remove: () => { obj.parent_object.remove(obj.root_object); }
    };

    const position = options.position ? options.position : new THREE.Vector3(0, 0, 0);
    const color = ValidateColor(options.color);
    const style = options.style ? options.style : 'normal';
    const size = options.size ? options.size : 1;
    const alignment = options.alignment ? options.alignment : 'left';
    const text = options.text ? options.text : options.text;
    const fontFamily = options.font_family ? options.font_family : 'Consolas, monospace, serif';
    const css_color = '#' + color.getHexString();

    const PosV3 = new THREE.Vector3(position.x, position.y, position.z);
    obj.root_object.position.set(PosV3.x, PosV3.y, PosV3.z);

    parent.add(obj.root_object);

    const text_instance = new THREE.TextSprite({
        alignment: alignment,
        color: css_color,
        fontFamily: fontFamily,
        fontStyle: style,
        fontSize: size,
        strokeWidth: 0,
        text: text,
    });

    obj.root_object.add(text_instance);
    return obj;
}

// Setup defaults scene and renderer
const DEBUG_SCENE_DEFAULTS = false;
function SetupSceneDefaults(viz) {
    const ambient_color = ValidateColor(viz.vizInfo.ambient_light_color, 0x404040);
    const ambient_intensity = (typeof viz.vizInfo.ambient_light_intensity === "number") ? viz.vizInfo.ambient_light_intensity : 0.4;

    // Set up scene ambient light
    const amb_light = new THREE.AmbientLight(ambient_color, ambient_intensity);
    viz.scene.add(amb_light);

    /*
    const top_light = new THREE.PointLight(0xffffff , 1.0, 100, 1.0);
    top_light.position.set(0, 15, 5);
    top_light.castShadow = true;
    top_light.shadow.mapSize.width = top_light.shadow.mapSize.height = shadowMapRes;
    viz.scene.add(top_light);

    const main_light = new THREE.PointLight(0xfeffd4 , 0.65, 100, 1);
    main_light.position.set(-15, -6, 15);
    main_light.castShadow = true;
    main_light.shadow.mapSize.width = main_light.shadow.mapSize.height = shadowMapRes;
    viz.scene.add(main_light);

    const soft_light = new THREE.PointLight(0xffffff , 0.35, 100, 1);
    soft_light.position.set(15, -6, 15);
    soft_light.castShadow = true;
    soft_light.shadow.mapSize.width = soft_light.shadow.mapSize.height = shadowMapRes;
    viz.scene.add(soft_light);
*/
    /*
    const planeGeometry2 = new THREE.PlaneGeometry( 10, 10, 32, 32 );
    const planeMaterial2 = new THREE.MeshStandardMaterial( { color: 0xB0B0B0,    transparent: true,
        opacity: 0.0001 } )
    planeMaterial2.side = THREE.DoubleSide;
    const plane2 = new THREE.Mesh( planeGeometry2, planeMaterial2 );
    plane2.rotation.x = 1.9;
    plane2.rotation.z = 0.75;
    plane2.position.y = 2.5;
    plane2.position.z = 0;
    plane2.receiveShadow = false;
    plane2.castShadow = true;
    //viz.scene.add( plane2 );

    const planeGeometry = new THREE.PlaneGeometry( 30, 30, 32, 32 );
    const planeMaterial = new THREE.MeshStandardMaterial( { color: 0xB0B0B0 } )
    planeMaterial.side = THREE.DoubleSide;
    plane = new THREE.Mesh( planeGeometry, planeMaterial );
    plane.rotation.x = 1.9;
    plane.position.y = -5;
    plane.receiveShadow = true;
    //viz.scene.add( plane );
    */

    //viz.camera.position.y = 10;
    viz.camera.position.z = 50;
    viz.camera.zoom = 1;
    console.log(viz.camera.rotation)
    viz.camera.updateProjectionMatrix();

    if (DEBUG_SCENE_DEFAULTS) {
        const geometry = new THREE.BoxGeometry();
        const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
        const cube = new THREE.Mesh(geometry, material);
        viz.scene.add(cube);

        const geometry2 = new THREE.BoxGeometry();
        const material2 = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
        const cube2 = new THREE.Mesh(geometry2, material2);
        cube2.position.x = cube2.position.y = 5;
        viz.scene.add(cube2);

        GenerateArrow(viz.scene, { x: 0, y: 0, z: 1 }, { x: 5, y: 5, z: 1 }, 0.5, 1.0, 1.0, 0x00ffff);
        GenerateText(viz.scene, { x: 0, y: 0, z: 1 }, 'Wow is that some text?', 1.5, 0xFF0000)

    }

    viz.render = () => {
        viz.preRenderCallbacks.forEach((fn) => {
            fn();
        });

        //requestAnimationFrame(animate);
        viz.renderer.render(viz.scene, viz.camera);

        viz.postRenderCallbacks.forEach((fn) => {
            fn();
        });
    }
    viz.render();
}

// Create viz object from viz info
async function CreateVizObject(vizInfo) {
    let obj = {
        scene: new THREE.Scene(),
        name: vizInfo.name,
        scenePixelWidth: vizInfo.pixel_width,
        scenePixelHeight: vizInfo.pixel_height,
        sceneOrthoUnitWidth: vizInfo.world_unit_width,
        sceneBgColor: vizInfo.background_color,
        saveImageThroughBrowser: vizInfo.save_through_browser,
        vizInfo: vizInfo,
        preRenderCallbacks: [],
        postRenderCallbacks: [],
        CreateIterationSpacePair: (options) => { return GenerateIterationSpacePair(obj, options); },
        CreateIterationSpace: (options) => { return GenerateIterationSpace(obj, options); },
        Create4dIterationSpace: (options) => { return Generate4dIterationSpace(obj, options); },
        Create5dIterationSpace: (options) => { return Generate4dIterationSpace(obj, options); },
        CreateAxisLabel: (options) => { return GenerateAxisLabel(obj, options); },
        CreateIterationSpaceLighting: (options) => { return GenerateIterationSpaceLighting(obj, options); },
        CreateLightingCube: (options) => { return GenerateLightingCube(obj, options); },
        CreateLightingRing: (options) => { return GenerateLightingRing(obj, options); },
        CreateArrow: (options) => { return GenerateArrow(obj, options); },
        CreateLine: (options) => { return GenerateLine(obj, options); },
        CreateText: (options) => { return GenerateText(obj, options); },
        SaveImage: (img_name) => {
            obj.render();
            return AutoSaveFrame(img_name, vizInfo);
        },
        CreateSequence: (sequence_name, fps) => { return CreateSequence(obj, sequence_name, fps); }
    }

    const normalized_width = (obj.scenePixelWidth > obj.scenePixelHeight) ? 1.0 : obj.scenePixelWidth / obj.scenePixelHeight;
    const normalized_height = (obj.scenePixelHeight > obj.scenePixelWidth) ? 1.0 : obj.scenePixelHeight / obj.scenePixelWidth;

    const ortho_width = obj.sceneOrthoUnitWidth * normalized_width;
    const ortho_height = obj.sceneOrthoUnitWidth * normalized_height;

    console.log("Setting up canvas with width: " + obj.scenePixelWidth + " and height: " + obj.scenePixelHeight + " will show " + obj.sceneOrthoUnitWidth + " world units at once")

    //obj.camera = new THREE.OrthographicCamera(ortho_width / - 2, ortho_width / 2, ortho_height / 2, ortho_height / - 2, 1, 1000 );
    const default_fov = 45;
    obj.camera = new THREE.PerspectiveCamera(default_fov, obj.scenePixelWidth / obj.scenePixelHeight, 1, 1000);
    obj.camera.set_fov_zoom = (zoom_factor) => {
        obj.camera.fov = default_fov * zoom_factor;
        obj.camera.updateProjectionMatrix();
    }

    obj.renderer = new THREE.WebGLRenderer({
        preserveDrawingBuffer: true,
        antialias: true,
        shadowMapType: THREE.PCFSoftShadowMap
    });
    obj.renderer.shadowMap.enabled = true;
    obj.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    obj.renderer.setSize(obj.scenePixelWidth, obj.scenePixelHeight);
    obj.renderer.setClearColor(new THREE.Color(obj.sceneBgColor));
    document.getElementById("render_container").appendChild(obj.renderer.domElement);

    let loader = new THREE.CubeTextureLoader();

    return obj;
}

/*
const color_scale_factor = 128 / shape[0];
const color_string = "0x" +  (128 + Math.imul(x, color_scale_factor)).toString(16) + (128 + Math.imul(y, color_scale_factor)).toString(16) + (128 + Math.imul(z, color_scale_factor)).toString(16);
*/