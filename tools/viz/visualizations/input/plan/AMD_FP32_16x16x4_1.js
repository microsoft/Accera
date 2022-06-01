////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

async function VizInfo() {
    const common = GetCommonConstants();

    return {
        name: "MFMA",
        pixel_width: common.pixel_width,
        pixel_height: common.pixel_height,
        background_color: common.background_color,
        ambient_light_color: common.ambient_light_color,
        ambient_light_intensity: common.ambient_light_intensity,
        shadow_map_resolution: common.shadow_map_resolution
    }
}

function LabelThread(Viz, text, size, color, position, flipped = false, x_tilt = 0.25, y_tilt = 0.8, length = 0.5) {
    const label1 = Viz.CreateText({
        position: position,
        text: text,
        size: size,
    });

    start_position = { x: position.x - x_tilt, y: position.y - y_tilt, z: position.z };
    end_position = { x: position.x, y: position.y - (y_tilt - length), z: position.z };

    if (flipped) {
        start_position = { x: position.x - x_tilt, y: position.y + (y_tilt - length), z: position.z };
        end_position = { x: position.x - (x_tilt * 2), y: position.y + ((y_tilt - length) * 2.5), z: position.z };
    }

    const line1 = Viz.CreateLine({
        start_position: start_position,
        end_position: end_position,
        line_thickness: 0.1,
        line_color: color,
        ignore_lighting: true
    });
}

async function RunViz(Viz, SceneView) {

    const common = GetCommonConstants();

    let d0 = 4;
    let d1 = 16;
    let d2 = 4;

    const space1 = Viz.CreateIterationSpace({
        shape: [d0, d1, d2],
        size: 1.0,
        padding: 0.15,
        position: { x: 0, y: 0, z: 0 },
        color: common.ball_color,
        rotation: { x: 0.5, y: 2.55, z: 3.1 }
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
            arrow_length: d1 - 0.5,
            label: "j",
            label_pos: 0.3,
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
            arrowhead_length: 1.2,
            arrow_length: d2,
            arrow_start_offset: 0,
            label: "ii",
            label_pos: 0.7,
            label_size: 1,
            arrow_distance_from_edge: 1,
            arrow_display_side: CUBE_SIDES.RIGHT,
        },
    });

    const lighting1 = Viz.CreateIterationSpaceLighting({
        space: space1,
        show_lights: false,
        top_light: { color: 0xffffff, panel_distance: 20, intensity: 0.6, distance: 35, decay: 2, light_count: { x: 2, y: 1 }, light_stride: { x: 12, y: 16 }, offset: { x: 0, y: -5 } },
        left_light: { color: 0xffffff, panel_distance: 15, intensity: 1, distance: 30, decay: 2, light_count: { x: 2, y: 2 }, light_stride: { x: 12, y: 10 }, offset: { x: 0, y: 8 } },
        right_light: { color: 0xffffff, panel_distance: 20, intensity: 1, distance: 40, decay: 2.2, light_count: { x: 2, y: 1 }, light_stride: { x: 18, y: 18 }, offset: { x: 3, y: 8 } },
    });

    Viz.camera.position.z = 45;
    Viz.camera.position.y = -1;
    Viz.camera.position.x = 0;
    Viz.camera.set_fov_zoom(0.6);

    const num_threads = 64;
    const colors = GetKnownColors(num_threads, true);

    // just color a subset for visbility
    for (let i = 0; i < 2; ++i) {
        for (let j = 0; j < 4; ++j) {
            for (let ii = 0; ii < d2; ++ii) {
                space1.set_child_color([i, j, ii], colors[i * d1 + j]);
            }
        }
    }

    // label some threads
    const label_text_size = 0.6
    i = 0;
    for (j = 0; j < 4; ++j) {
        LabelThread(Viz, "t" + j, label_text_size, colors[i * d1 + j], { x: -5 + (0.8 * j), y: 5 - (0.15 * j), z: 0 });
    }

    // color and label the last thread on the first row
    i = 0;
    j = d1 - 1;
    for (ii = 0; ii < d2; ++ii) {
        space1.set_child_color([i, j, ii], colors[i * d1 + j]);
    }
    LabelThread(Viz, "t" + j, label_text_size, colors[i * d1 + j], { x: 9.4, y: 1.8, z: 0 });

    // color and label the last thread on the last row
    i = d0 - 1;
    j = d1 - 1;
    for (ii = 0; ii < d2; ++ii) {
        space1.set_child_color([i, j, ii], colors[i * d1 + j]);
    }

    LabelThread(Viz, "t" + (num_threads - 1), label_text_size, colors[num_threads - 1], { x: 7.5, y: -6, z: 0 }, flipped = true);

    await Viz.SaveImage("AMD_FP32_16x16x4_1");
}

