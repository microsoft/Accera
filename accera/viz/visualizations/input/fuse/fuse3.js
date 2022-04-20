////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

async function VizInfo() {
    const common = GetCommonConstants();
    
    return {
        name: "Fuse",
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
    console.log(GetCommonConstants());
    const dummy_lighting = Viz.CreateIterationSpaceLighting({
        space: dummy_space,
        show_lights: false,
        top_light: {color: 0xffffff, panel_distance: 8, intensity : 0.5, distance : 35, decay : 2,  light_count: {x: 1, y: 1}, light_stride: {x: 12, y: 16}, offset: {x: 5, y: 0}},
        left_light: {color: 0xffffff, panel_distance: 12, intensity : 1.25, distance : 30, decay : 2,  light_count: {x: 2, y: 2}, light_stride: {x: 12, y: 10}, offset: {x: 20, y: 0}},
        right_light: {color: 0xffffff, panel_distance: 30, intensity : 0.05, distance : 40, decay : 2.2,  light_count: {x: 2, y: 2}, light_stride: {x: 18, y: 18}, offset: {x: 0, y: 10}},
   });

    dummy_lighting.root_object.position.x -= 15;
    dummy_space.remove();

    return dummy_lighting;
}

async function RunViz(Viz, SceneView) {
    const common = GetCommonConstants(); 

    let d0 = 4;
    let d1 = 6;
    let d2 = 5;
    let d3 = 10;

    const space1_params = {
        shape: [d0,d1,d2],
        size: 1.0,
        padding: 0.15,
        position: {x: 0, y:0, z:0},
        color: common.fuse_color1,
        rotation:  {x :0.4, y: 2.7, z: 3.1}
    };

    const space2_params = {
        shape: [d0,d3,d1],
        size: 1.0,
        padding: 0.15,
        position: {x: 0, y:0, z:0},
        color: common.fuse_color2,
        rotation:  {x :0.4, y: 2.5, z: 3.1}
    };

    let space_1_label = {
        axis0: {
            color: common.arrow_color, 
            arrow_thickness: 0.25, 
            arrowhead_length: 1.0,
            arrowhead_thickness: 0.5, 
            arrowhead_length: 1, 
            arrow_length: d0, 
            arrow_display_side: CUBE_SIDES.LEFT,
            arrow_alignment_side: CUBE_SIDES.FRONT,
            arrow_distance_from_edge: 1,
            arrow_start_offset: 0,
            label: "i0", 
            label_pos: 0.5, 
            label_size: 1
        },
        axis1: {
            color: common.arrow_color, 
            arrow_thickness: 0.25, 
            arrowhead_length: 1.0,
            arrowhead_thickness: 0.5, 
            arrowhead_length: 1, 
            arrow_length: d1-1.5, 
            arrow_display_side: CUBE_SIDES.BOTTOM,
            arrow_alignment_side: CUBE_SIDES.FRONT,
            arrow_distance_from_edge: 1,
            arrow_start_offset: 0,
            label: "j0", 
            label_pos: 0.5, 
            label_size: 1
        },
        axis2: {
            color: common.arrow_color, 
            arrow_thickness: 0.25, 
            arrowhead_length: 1.0,
            arrowhead_thickness: 0.5, 
            arrowhead_length: 1.4, 
            arrow_length: d2, 
            //arrow_display_side: CUBE_SIDES.TOP,
            arrow_alignment_side: CUBE_SIDES.BACK,
            arrow_distance_from_edge: 1,
            arrow_start_offset: 0,
            label: "k0", 
            label_pos: 0.5, 
            label_size: 1
        },
    }

    let space_2_label = {
        axis0: {
            color: common.arrow_color, 
            arrow_thickness: 0.25, 
            arrowhead_length: 1.0,
            arrowhead_thickness: 0.5, 
            arrowhead_length: 1, 
            arrow_length: d0, 
            arrow_display_side: CUBE_SIDES.LEFT,
            arrow_alignment_side: CUBE_SIDES.FRONT,
            arrow_distance_from_edge: 1,
            arrow_start_offset: 0,
            label: "i1", 
            label_pos: 0.5, 
            label_size: 1
        },
        axis1: {
            color: common.arrow_color, 
            arrow_thickness: 0.25, 
            arrowhead_length: 1.0,
            arrowhead_thickness: 0.5, 
            arrowhead_length: 1, 
            arrow_length: d3-1, 
            arrow_display_side: CUBE_SIDES.BOTTOM,
            arrow_alignment_side: CUBE_SIDES.FRONT,
            arrow_distance_from_edge: 1,
            arrow_start_offset: 0,
            label: "j1", 
            label_pos: 0.5, 
            label_size: 1
        },
        axis2: {
            color: common.arrow_color, 
            arrow_thickness: 0.25, 
            arrowhead_length: 1.0,
            arrowhead_thickness: 0.5, 
            arrowhead_length: 1.4, 
            arrow_length: d1, 
            //arrow_display_side: CUBE_SIDES.TOP,
            arrow_alignment_side: CUBE_SIDES.BACK,
            arrow_distance_from_edge: 1,
            arrow_start_offset: 0,
            label: "  k1", 
            label_pos: 0.5, 
            label_size: 1
        },    
    }


    let space_pair = Viz.CreateIterationSpacePair({
        space1_params: space1_params,
        space2_params: space2_params,
        space1_axis_label: space_1_label,
        space2_axis_label: space_2_label,
        spacing: 4,
        horizontal: true
    });

    Viz.camera.position.z = 45;
    Viz.camera.position.y = -1;
    Viz.camera.position.x = 6.5;
    Viz.camera.set_fov_zoom(0.6);

    let lighting = GenerateLightingGrid(Viz);

    await Viz.SaveImage("fuse3")
}

