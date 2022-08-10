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

async function RunViz(Viz, SceneView) {

    const common = GetCommonConstants();

    let d0 = 3;
    let d1 = 12;
    let d2 = 15;

    const space1 = Viz.CreateIterationSpace({
        shape: [d0, d1, d2],
        size: 1.0,
        padding: 0.3,
        position: { x: 0, y: 0, z: 0 },
        color: common.ball_color,
        rotation: { x: 0.4, y: 2.65, z: 3.1 }
    });

    const axis1 = Viz.CreateAxisLabel({
        space: space1,
        axis0: {
            color: common.arrow_color,
            arrow_thickness: 0.25,
            arrowhead_length: 1.0,
            arrowhead_thickness: 0.5,
            arrowhead_length: 1,
            arrow_length: d0 - 0.5,
            arrow_display_side: CUBE_SIDES.LEFT,
            arrow_alignment_side: CUBE_SIDES.FRONT,
            arrow_distance_from_edge: 1,
            arrow_start_offset: 0,
            label: "i",
            label_pos: 0.5,
            label_size: 1
        },
        axis1: {
            color: common.arrow_color,
            arrow_thickness: 0.25,
            arrowhead_thickness: 0.5,
            arrowhead_length: 1,
            arrow_length: d1 + 1,
            label: "j",
            label_pos: 0.1,
            label_size: 1,
            arrow_start_offset: 0,
            arrow_distance_from_edge: 1,
            arrow_display_side: CUBE_SIDES.BOTTOM,
            arrow_alignment_side: CUBE_SIDES.FRONT,
        },
        axis2: {
            color: common.arrow_color,
            arrow_thickness: 0.25,
            arrowhead_thickness: 0.5,
            arrowhead_length: 2,
            arrow_length: d2 + 1,
            arrow_start_offset: 0,
            label: "k",
            label_pos: 0.1,
            label_size: 1,
            arrow_distance_from_edge: 1,
            arrow_display_side: CUBE_SIDES.RIGHT,
        },
    });


    const lighting1 = Viz.CreateIterationSpaceLighting({
        space: space1,
        show_lights: false,
        top_light: { color: 0xffffff, panel_distance: 20, intensity: 0.6, distance: 35, decay: 2, light_count: { x: 2, y: 2 }, light_stride: { x: 12, y: 16 }, offset: { x: 0, y: -5 } },
        left_light: { color: 0xffffff, panel_distance: 12, intensity: 1.2, distance: 30, decay: 2, light_count: { x: 2, y: 1 }, light_stride: { x: 12, y: 10 }, offset: { x: 0, y: 8 } },
        right_light: { color: 0xffffff, panel_distance: 20, intensity: 1, distance: 40, decay: 2.2, light_count: { x: 2, y: 1 }, light_stride: { x: 18, y: 18 }, offset: { x: 3, y: 8 } },
    });

    Viz.camera.position.z = 45;
    Viz.camera.position.y = 0;
    Viz.camera.position.x = -2;
    Viz.camera.set_fov_zoom(0.6);

    await Viz.SaveImage("space_3_12_15_presplit");
}

