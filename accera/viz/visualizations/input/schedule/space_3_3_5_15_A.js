////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

async function VizInfo() {
    const common = GetCommonConstants();
    
    return {
        name: "Schedule",
        pixel_width: common.pixel_width,
        pixel_height: common.pixel_height,
        background_color: common.background_color,
        ambient_light_color: common.ambient_light_color,
        ambient_light_intensity: common.ambient_light_intensity,
        shadow_map_resolution: common.shadow_map_resolution
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

    const dummy_lighting = Viz.CreateIterationSpaceLighting({
        space: dummy_space,
        show_lights: false,
        top_light: {color: 0xffffff, panel_distance: -4, intensity : 0.8, distance : 35, decay : 2,  light_count: {x: 2, y: 2}, light_stride: {x: 12, y: 16}, offset: {x: 0, y: -5}},
        left_light: {color: 0xffffff, panel_distance: 13, intensity : 1.1, distance : 30, decay : 2,  light_count: {x: 2, y:1}, light_stride: {x: 12, y: 5}, offset: {x: 0, y: 8}},
        right_light: {color: 0xffffff, panel_distance: 20, intensity : 1, distance : 40, decay : 2.2,  light_count: {x: 2, y: 1}, light_stride: {x: 18, y: 18}, offset: {x: 3, y: 8}},
    });

    //dummy_lighting.root_object.position.x -= 15;
    dummy_space.remove();

    return dummy_lighting;
}

async function RunViz(Viz, SceneView) {
    const common = GetCommonConstants(); 

    const default_size = 1.0;
    const default_padding = 0.14;
    const default_rotation = {x :0.6, y: 2.65, z: 3.1};
    const split_size = 3;

    // Dimensions of each space in the higher dimensional space
    let d0 = 3;
    let d1 = 5;
    let d2 = 15;

    Viz.camera.position.z = 45;
    Viz.camera.position.y = -1;
    Viz.camera.position.x = -1;
    Viz.camera.set_fov_zoom(0.6);

    let split_space_params = {
        shape: [d0, d1, d2],
        size: default_size,
        padding: default_padding,
        position: {x: 0, y:0, z:0},
        color: common.ball_color,
        rotation: GetDefault2dIterationPose()
    };

    let split_axis_labels = {
        axis0: {
            color: common.arrow_color, 
            arrow_thickness: 0.25, 
            arrowhead_length: 1.0,
            arrowhead_thickness: 0.5, 
            arrowhead_length: 1, 
            arrow_length: d0-0.5, 
            arrow_display_side: CUBE_SIDES.LEFT,
            arrow_alignment_side: CUBE_SIDES.FRONT,
            arrow_distance_from_edge: 0.75,
            arrow_start_offset: -0.3,
            label: " j", 
            label_pos: 0.7, 
            label_size: 1
        },
        axis1: {
            color: common.arrow_color, 
            arrow_thickness: 0.25, 
            arrowhead_thickness: 0.5, 
            arrowhead_length: 1, 
            arrow_length: d1-0.5, 
            label: "jj", 
            label_pos: 0.3, 
            label_size: 1,
            arrow_start_offset: -0.3,
            arrow_distance_from_edge: 0.75,
            arrow_display_side: CUBE_SIDES.BOTTOM,
            arrow_alignment_side: CUBE_SIDES.FRONT,
        },
        axis2: {
            color: common.arrow_color, 
            arrow_thickness: 0.25, 
            arrowhead_thickness: 0.5, 
            arrowhead_length: 2, 
            arrow_length: d2, 
            arrow_start_offset: -1,
            label: "k ", 
            label_pos: 0.3, 
            label_size: 1,
            arrow_distance_from_edge: 1,
            arrow_display_side: CUBE_SIDES.RIGHT,
        },
    }

    let lighting = GenerateLightingGrid(Viz);

    const space5d = Viz.Create5dIterationSpace({
        inner_space_params: split_space_params,
        inner_space_axis_labels: split_axis_labels,
        inner_space_label_coords: [[0, 1]],
        horizontal_spacing: 2.5,
        vertical_spacing: 2,
        position: {x: 0, y:0, z:0},
        rotation: {x: 0, y: 0, z:0},
        outer_axis1_size: 3,
        outer_axis1: {
            color: common.arrow_color, 
            arrow_thickness: 0.3, 
            arrowhead_thickness: 0.6, 
            arrowhead_length: 1, 
            arrow_display_side: CUBE_SIDES.TOP,
            arrow_alignment_side: CUBE_SIDES.BACK,
            arrow_distance_from_edge: 1,
            arrow_start_offset: 1,
            arrow_length: 20, 
            label: "i", 
            label_pos: 0.3, 
            label_size: 1.25,
            // label_font: '"Times New Roman", Times, serif'
        },
    });

    // We can attach the lighting to the iteration space so it rotates with us.
    lighting.root_object.rotation.x = 0;
    lighting.root_object.rotation.y = 0;
    space5d.root_object.attach(lighting.root_object);

    // Rotate iteration space for a nicer view
    space5d.root_object.rotation.x = 0.6;
    space5d.root_object.rotation.y = -0.3;
    space5d.root_object.rotation.z = 0.02;


    for(let s = 0; s < 3; ++s) {
        for(let jj = 2; jj < 5; ++jj) {
            for(let k = 0; k < d2; ++k) {
                space5d.get_iteration_space([0, s]).set_child_color(
                    [2,jj,k],
                    common.padding_color1
                );
            }
        }
    }

    await Viz.SaveImage("space_3_3_5_15_A");
}

