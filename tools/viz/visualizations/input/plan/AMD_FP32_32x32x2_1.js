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

function LabelThread(Viz, text, size, color, position, flipped = false, length = 0.5, x_tilt = 0.25, y_tilt = 0.8) {
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

function GenerateLightingGrid(Viz) {
    const dummy_space_params = {
        shape: [10, 40, 10],
        size: 1.0,
        padding: 0.32,
        position: { x: 0, y: 0, z: 0 },
        color: 0x808080,
        rotation: GetDefault2dIterationPose()
    };
    const dummy_space = Viz.CreateIterationSpace(dummy_space_params);

    const dummy_lighting = Viz.CreateIterationSpaceLighting({
        space: dummy_space,
        show_lights: false,
        top_light: { color: 0xFFFFFF, panel_distance: 14, intensity: 1.1, distance: 55, decay: 2, light_count: { x: 1, y: 1 }, light_stride: { x: 30, y: 30 }, offset: { x: -1, y: 0 } },
        left_light: { color: 0xFFFFFF, panel_distance: 12, intensity: 1, distance: 55, decay: 2, light_count: { x: 2, y: 2 }, light_stride: { x: 30, y: 30 }, offset: { x: 0, y: -3 } },
        right_light: { color: 0xFFFFFF, panel_distance: 30, intensity: 0.55, distance: 55, decay: 2.2, light_count: { x: 2, y: 2 }, light_stride: { x: 30, y: 30 }, offset: { x: 0, y: -3 } },
    });

    dummy_lighting.root_object.position.x -= 15;
    dummy_space.remove();

    return dummy_lighting;
}

async function RunViz(Viz, SceneView) {

    const common = GetCommonConstants();

    let d0 = 4;
    let d1 = 32;
    let d2 = 4;
    let split_size = 2;

    let split_space_params = {
        shape: [d0, d1, d2],
        size: 1.0,
        padding: 0.15,
        position: { x: 0, y: 0, z: 0 },
        color: common.ball_color,
        rotation: { x: 0.5, y: 2.55, z: 3.1 }
    };

    let split_axis_labels = {
        axis0: {
            color: common.arrow_color,
            arrow_thickness: 0.25,
            arrowhead_length: 1.0,
            arrowhead_thickness: 0.5,
            arrowhead_length: 1,
            arrow_length: d0 - 0.5,
            arrow_display_side: CUBE_SIDES.LEFT,
            arrow_alignment_side: CUBE_SIDES.FRONT,
            arrow_distance_from_edge: 1.8,
            arrow_start_offset: 0,
            label: "ii",
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
            label: "iii",
            label_pos: 0.3,
            label_size: 1,
            arrow_distance_from_edge: 1,
            arrow_display_side: CUBE_SIDES.RIGHT,
        },
    };

    Viz.camera.position.z = 45;
    Viz.camera.position.y = -1;
    Viz.camera.position.x = 0;
    Viz.camera.set_fov_zoom(0.98);

    const space5d = Viz.Create5dIterationSpace({
        inner_space_params: split_space_params,
        inner_space_axis_labels: split_axis_labels,
        horizontal_spacing: 0,
        vertical_spacing: 5,
        position: { x: 0, y: 0, z: 0 },
        rotation: { x: 0, y: 0, z: 0 },
        outer_axis0_size: split_size,
        outer_axis0: {
            color: common.arrow_color,
            arrow_thickness: 0.25,
            arrowhead_thickness: 0.5,
            arrowhead_length: 1.2,
            arrow_display_side: CUBE_SIDES.LEFT,
            arrow_alignment_side: CUBE_SIDES.BACK,
            arrow_distance_from_edge: 0,
            arrow_start_offset: 1,
            arrow_length: 8,
            label: "i",
            label_pos: 0.1,
            label_size: 1.25,
        },
    });

    let lighting = GenerateLightingGrid(Viz);
    space5d.root_object.attach(lighting.root_object);

    const num_threads = 64;
    const colors = GetKnownColors(num_threads, shuffle = true);
    const color_offset = num_threads / 2;

    // thread assignments: i = 0 => t0-31; i = 1 => t32-63
    // label some threads
    const label_text_size = 0.6
    let i = 0;
    let ii = 0;
    for (let j = 0; j < 4; ++j) { // t0-3
        LabelThread(Viz,
            "t" + j,
            label_text_size,
            colors[ii * d1 + j],
            { x: -11.2 + (0.8 * j), y: 10.3 - (0.15 * j), z: 0 });
    }
    j = d1 - 1; // t31
    LabelThread(Viz,
        "t" + j,
        label_text_size,
        colors[ii * d1 + j],
        { x: 20.3, y: 5.2, z: 0 });

    i = 1;
    for (j = 0; j < 4; ++j) { // t32-35
        LabelThread(Viz,
            "t" + (color_offset + j),
            label_text_size * 0.8,
            colors[color_offset + ii * d1 + j],
            { x: -10.9 + (0.8 * j), y: 2.4 - (0.3 * j), z: 0 });
    }
    j = d1 - 1; // t63
    LabelThread(Viz,
        "t" + (color_offset + j),
        label_text_size,
        colors[color_offset + ii * d1 + j],
        { x: 20.3, y: -6.8, z: 0 });

    // color some threads
    for (ii = 0; ii < d0; ++ii) {
        for (i = 0; i < split_size; ++i) {
            for (j = 0; j < 4; ++j) {
                for (let iii = 0; iii < d2; ++iii) {
                    space5d.get_iteration_space([0, i]).set_child_color(
                        [ii, j, iii],
                        colors[(i * color_offset + j)]
                    );
                }
            }
            for (j = d1 - 2; j < d1; ++j) {
                for (let iii = 0; iii < d2; ++iii) {
                    space5d.get_iteration_space([0, i]).set_child_color(
                        [ii, j, iii],
                        colors[(i * color_offset + j)]
                    );
                }
            }
        }
        await Viz.SaveImage("AMD_FP32_32x32x2_1_" + ii);
    }
}
