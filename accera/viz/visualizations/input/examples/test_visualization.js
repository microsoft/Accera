////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

async function VizInfo() {
    return {
        name: "Test Viz",           // Name for our visualization
        pixel_width: 1600,          // How many pixels wide our viewport is
        pixel_height: 900,          // How many pixels tall our viewport is
        world_unit_width: 120,       // How many world units wide our viewport is
        background_color: 0xFFFFFFFF// What color should the background of the scene be
    }
}

async function RunViz(Viz, SceneView) {
    const space1 = Viz.CreateIterationSpace({shape: [19, 16, 13],
                                            size: 1.0,
                                            padding: 0.4,
                                            position: {x: 0, y:0, z:0},
                                            color: 0x808080,
                                            rotation: GetDefault3dIterationPose()
    });

    const axis1 = Viz.CreateAxisLabel({
        space: space1,
        axis0: {color: 0x000000, arrow_thickness: 0.25, arrowhead_thickness: 0.5, arrow_length: 7, label: "i", label_pos: 2, label_size: 1.5},
        axis1: {color: 0x000000, arrow_thickness: 0.25, arrowhead_thickness: 0.5, arrow_length: 7, label: "j", label_pos: 2, label_size: 1.5},
        axis2: {color: 0x000000, arrow_thickness: 0.25, arrowhead_thickness: 0.5, arrow_length: 7, label: "k", label_pos: 2, label_size: 1.5}
    });

    const lighting1 = Viz.CreateIterationSpaceLighting({
        space: space1,
        top_light: {color: 0xFFFFFF, panel_distance: 15, intensity : 0.85, distance : 30, decay : 2,  light_count: {x: 2, y: 2}, light_stride: {x: 12, y: 16}, offset: {x: 0, y: 0}},
        left_light: {color: 0xfeffd4, panel_distance: 10, intensity : 1, distance : 30, decay : 2,  light_count: {x: 2, y: 2}, light_stride: {x: 12, y: 18}, offset: {x: 0, y: 0}},
        right_light: {color: 0xc7f8ff, panel_distance: 10, intensity : 0.55, distance : 30, decay : 2.2,  light_count: {x: 2, y: 2}, light_stride: {x: 12, y: 18}, offset: {x: 0, y: 0}},
    });

    // X = Red
    space1.set_child_color(
        {i: 1, j: 0, k: 0},                     // Which child to set color for
        0xFF0000 // Which color to set
    );

    // Y = Green
    space1.set_child_color(
        {i: 0, j: 1, k: 0},                     // Which child to set color for
        0x00FF00 // Which color to set
    );

    // Z = Blue
    space1.set_child_color(
        {i: 0, j: 0, k: 1},                     // Which child to set color for
        0x0000FF // Which color to set
    );

    await Viz.SaveImage("OneIterationSpace"); // Save a frame
    return;
    //const space2 = Viz.CreateIterationSpace([5, 5, 5], 1.0, 0.5, {x: space1.render_width*1.5, y:0, z:0}, 0x808080);
    //const space3 = Viz.CreateIterationSpace([5, 5, 5], 1.0, 0.5, {x: -space1.render_width*1.5, y:0, z:0}, 0x808080);

    await Viz.SaveImage("MultipleIterationSpaces");

    // Color each space based on x coordinate
    const stride_colors = [0xB565A7, 0x009B77, 0xDD4124, 0xD65076, 0x45B8AC];
    for (let x = 0; x < 5; ++ x)
    {
        for (let y = 0; y < 5; ++ y)
        {
            for (let z = 0; z < 5; ++ z)
            {
                space1.set_child_color(
                    {x: x, y: y, z: z},                     // Which child to set color for
                    stride_colors[x % stride_colors.length] // Which color to set
                );
continue;
                space2.set_child_color(
                    {x: x, y: y, z: z},                     // Which child to set color for
                    stride_colors[x % stride_colors.length] // Which color to set
                );
                space3.set_child_color(
                    {x: x, y: y, z: z},                     // Which child to set color for
                    stride_colors[x % stride_colors.length] // Which color to set
                );
            }
        }
    }

    await Viz.SaveImage("BeautifulIterationSpaces");
    return;
    Viz.CreateArrow(
        {x: -(space1.render_width/2), y: (space1.render_height/2)}, // Arrow head position
        {x: -(space1.render_width), y: (space1.render_height)},     // Arrow tail position
        0.05,                                                        // Arrow line thickness
        0.2,                                                        // Arrow head length
        0.2,                                                        // Arrow head thickness
        0x00ffff,                                                   // Arrow line color
        0x00ff00,                                                   // Arrow head color
        true                                                       // Should arrow ignore lighting?
    );

    Viz.CreateText(
        {x: -(space1.render_width/2), y: (space1.render_height) + 2}, // Text position
        'Wow is that an iteration space?',                            // Text
         1.2,                                                         // Text size
         0x000000,                                                    // Text color
         'italic',                                                    // Text style
         'left'                                                       // Text alignment
    )

    await Viz.SaveImage("IterationSpacesNowWithLabels");
    return;
    // Color each space based on x coordinate
    for (let x = 0; x < 5; ++ x)
    {
        for (let y = 0; y < 5; ++ y)
        {
            for (let z = 0; z < 5; ++ z)
            {
                space1.set_child_color(
                    {x: x, y: y, z: z},                     // Which child to set color for
                    stride_colors[x % stride_colors.length], // Which color to set
                    0.1
                );
            }
        }
        await Viz.SaveImage("Transparency" + x);
    }
}

