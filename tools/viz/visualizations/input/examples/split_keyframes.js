////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

async function VizInfo() {
    return {
        name: "Split Keyframes",           // Name for our visualization
        pixel_width: 1600,          // How many pixels wide our viewport is
        pixel_height: 900,          // How many pixels tall our viewport is
        world_unit_width: 120,       // How many world units wide our viewport is
        background_color: 0xFFFFFFFF,// What color should the background of the scene be
        shadow_map_resolution: 512
    }
}

function GenerateLightingGrid(Viz) {
    const dummy_space_params = {
        shape: [10, 40, 10],
        size: 1.0,
        padding: 0.32,
        position: {x: 0, y:0, z:0},
        color: 0x808080,
        rotation:  GetDefault2dIterationPose()
    };
    const dummy_space = Viz.CreateIterationSpace(dummy_space_params);
    console.log(GetCommonConstants());
    const dummy_lighting = Viz.CreateIterationSpaceLighting({
        space: dummy_space,
        show_lights: false,
        top_light: {color: 0xCCCCFF, panel_distance: 14, intensity : 1.7, distance : 55, decay : 2,  light_count: {x: 2, y: 2}, light_stride: {x: 30, y: 30}, offset: {x: -1, y: 0}},
        left_light: {color: 0xfeffd4, panel_distance: 12, intensity : 1.35, distance : 55, decay : 2,  light_count: {x: 2, y: 2}, light_stride: {x: 30, y: 30}, offset: {x: 0, y: -10}},
        right_light: {color: 0xc7f8ff, panel_distance: 30, intensity : 0.55, distance : 55, decay : 2.2,  light_count: {x: 2, y: 2}, light_stride: {x: 30, y: 30}, offset: {x: 0, y: 0}},
    });

    dummy_lighting.root_object.position.x -= 15;
    dummy_space.remove();

    return dummy_lighting;
}

async function RunViz(Viz, SceneView) {
    const default_size = 2.0;
    const default_padding = 0.25;
    const default_rotation = {x :0.6, y: 2.65, z: 3.1};
    const split_size = 3;

    // Dimensions of each space in the higher dimensional space
    let d0 = 4;
    let d1 = 3;
    let d2 = 9;

    Viz.camera.position.z = 75;
    Viz.camera.position.y = 0;
    Viz.camera.position.x = 0;


    const space1_params = {
        shape: [2, 8, 1],
        size: 2.0,
        padding: 0.25,
        position: {x: 0, y:0, z:0},
        color: 0x808080,
        rotation:  GetDefault2dIterationPose()
    };

    const space2_params = {
        shape: [8, 8, 6],
        size: 2.0,
        padding: 0.25,
        position: {x: 0, y:0, z:0},
        color: 0x808080,
        rotation:  GetDefault2dIterationPose()
    };

    let space_1_label = {
        axis0: {
            color: 0x000000, 
            arrow_thickness: 0.25, 
            arrowhead_length: 1.0,
            arrowhead_thickness: 0.5, 
            arrow_length: 3, 
            arrow_display_side: CUBE_SIDES.RIGHT,
            arrow_alignment_side: CUBE_SIDES.BACK,
            arrow_distance_from_edge: 2,
            arrow_start_offset: 0,
            label: "j", 
            label_pos: 1, 
            label_size: 1.5
        },
        axis1: {
            color: 0x000000, arrow_thickness: 0.25, arrowhead_thickness: 0.5, arrow_length: 3, label: "jj", label_pos: 1, label_size: 1.5,
            arrow_distance_from_edge: 2,
            arrow_display_side: CUBE_SIDES.BOTTOM,
            arrow_alignment_side: CUBE_SIDES.BACK,
        },
    }

    let space_2_label = {
        axis0: {
            color: 0x000000, 
            arrow_thickness: 0.25, 
            arrowhead_length: 1.0,
            arrowhead_thickness: 0.5, 
            arrow_length: 3, 
            arrow_display_side: CUBE_SIDES.RIGHT,
            arrow_alignment_side: CUBE_SIDES.BACK,
            arrow_distance_from_edge: 2,
            arrow_start_offset: 0,
            label: "j", 
            label_pos: 1, 
            label_size: 1.5
        },
        axis1: {
            color: 0x000000, arrow_thickness: 0.25, arrowhead_thickness: 0.5, arrow_length: 3, label: "jj", label_pos: 1, label_size: 1.5,
            arrow_display_side: CUBE_SIDES.BOTTOM,
            arrow_alignment_side: CUBE_SIDES.BACK,
        },
        axis2: {
            color: 0x000000, arrow_thickness: 0.25, arrowhead_thickness: 0.5, arrow_length: 3, label: "k", label_pos: 1, label_size: 1.5,
            arrow_display_side: CUBE_SIDES.RIGHT,
        },
    }


    let space_pair = Viz.CreateIterationSpacePair({
        space1_params: space1_params,
        space2_params: space2_params,
        space1_axis_label: space_1_label,
        space2_axis_label: space_2_label,
        spacing: 0,
        horizontal: false
    });

    let lighting = GenerateLightingGrid(Viz);

    await Viz.SaveImage("test")

    space_pair.root_object.rotation.x = .75;

    await Viz.SaveImage("test2")

    space_pair.remove();

    let split_space_params = {
        shape: [d0, d1, d2],
        size: default_size,
        padding: default_padding,
        position: {x: 0, y:0, z:0},
        color: 0x808080,
        rotation: GetDefault2dIterationPose()
    };

    let split_axis_labels = {
        axis0: {
            color: 0x000000, 
            arrow_thickness: 0.25, 
            arrowhead_length: 1.0,
            arrowhead_thickness: 0.5, 
            arrow_length: 3, 
            arrow_display_side: CUBE_SIDES.LEFT,
            arrow_alignment_side: CUBE_SIDES.BACK,
            arrow_distance_from_edge: 1,
            arrow_start_offset: 0,
            label: "j", 
            label_pos: 1, 
            label_size: 1.5
        },
        axis1: {
            color: 0x000000, arrow_thickness: 0.25, arrowhead_thickness: 0.5, arrow_length: 3, label: "jj", label_pos: 1, label_size: 1.5,
            arrow_display_side: CUBE_SIDES.BOTTOM,
            arrow_alignment_side: CUBE_SIDES.BACK,
        },
        axis2: {
            color: 0x000000, arrow_thickness: 0.25, arrowhead_thickness: 0.5, arrow_length: 3, label: "k", label_pos: 1, label_size: 1.5,
            arrow_display_side: CUBE_SIDES.RIGHT,
        },
    }


    const space4d = Viz.Create4dIterationSpace({
        inner_space_params: split_space_params, // The parameters used to create each space inside the 4d array
        inner_space_axis_labels: split_axis_labels, // The axis labels for each space inside the 4d array, omit to not have labels
        inner_space_label_coords: [[0], [2]], // Which inner spaces should be labeled, omit to label all of the spaces
        vertical_spacing: 7,
        position: {x: 0, y:0, z:0}, // Position of the overall 4d array
        rotation: {x: 0, y: 0, z:0}, // Rotation of the overall 4d array
        outer_axis0_size: 3, // The outer axis size which runs along the "top" of the 4d array
        outer_axis0: {
            color: 0x000000, arrow_thickness: 0.25, arrowhead_thickness: 0.5, arrowhead_length: 1.5, arrow_length: 20, label: "i", label_pos: 1, label_size: 1.5,
            arrow_distance_from_edge: 0,
            arrow_display_side: CUBE_SIDES.LEFT,
            arrow_alignment_side: CUBE_SIDES.FRONT
        } // The outer axis label which runs along the "top" of the 4d array
    });

    await Viz.SaveImage("4dSplitSpace")

    split_axis_labels.axis0.label = "<3";
    space4d.get_iteration_space_label([0]).set_axis_params(split_axis_labels);

    await Viz.SaveImage("4dSplitSpaceChanged")

    // Rotate iteration space for a nicer view
    space4d.root_object.rotation.x = 0.5;
    space4d.root_object.rotation.y = -0.5;

    await Viz.SaveImage("4dSplitSpaceRotated")

    const space5d_params = {
        inner_space_params: split_space_params,
        horizontal_spacing: 0,
        vertical_spacing: 0,
        position: {x: 0, y:0, z:0},
        rotation: {x: 0, y: 0, z:0},
        outer_axis0_size: 3,
        outer_axis0: {
            color: 0x000000, 
            arrow_thickness: 0.25, 
            arrowhead_thickness: 0.5, 
            arrowhead_length: 1.5, 
            arrow_display_side: CUBE_SIDES.LEFT,
            arrow_alignment_side: CUBE_SIDES.FRONT,
            arrow_length: 20, 
            label: "i", 
            label_pos: 0, 
            label_size: 1.5,
            label_font: '"Times New Roman", Times, serif'
        },
        outer_axis1_size: 5,
        outer_axis1: {
            color: 0x000000, 
            arrow_thickness: 0.25, 
            arrowhead_thickness: 0.5, 
            arrowhead_length: 1.5, 
            arrow_length: 20, 
            arrow_display_side: CUBE_SIDES.BOTTOM,
            arrow_alignment_side: CUBE_SIDES.FRONT,
            label: "ii", 
            label_pos: 0.5,
            label_size: 1.5
        },
    };
    const space5d = Viz.Create5dIterationSpace(space5d_params);

    // We can attach the lighting to the iteration space so it rotates with us.
    lighting.root_object.rotation.x = 0;
    lighting.root_object.rotation.y = 0;
    space5d.root_object.attach(lighting.root_object);

    // Remove the old space
    space4d.remove();

    space5d.get_iteration_space([1, 2]).set_child_color([1, 2, 0], 0xFF0000);

    await Viz.SaveImage("5dSplitSpace")

    space5d.get_iteration_space_label([0, 2]).remove();
    space5d_params.outer_axis0.color = 0xFF0000;
    space5d_params.outer_axis0.label = '(>")>';
    space5d.axis1.remove();
    space5d.set_axis0_params(space5d_params.outer_axis0);

    await Viz.SaveImage("5dChangeLabels")

    // Rotate iteration space for a nicer view
    space5d.root_object.rotation.x = 0.5;
    space5d.root_object.rotation.y = -0.5;

    await Viz.SaveImage("5dSplitSpaceRotated")

    space5d.remove_all_axis_labels();
    await Viz.SaveImage("5dSplitSpaceNoLabels")

    return;
}

