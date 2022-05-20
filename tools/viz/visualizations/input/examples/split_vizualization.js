////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

async function VizInfo() {
    return {
        name: "Split Viz",           // Name for our visualization
        pixel_width: 1600,          // How many pixels wide our viewport is
        pixel_height: 900,          // How many pixels tall our viewport is
        world_unit_width: 120,       // How many world units wide our viewport is
        background_color: 0xFFFFFFFF// What color should the background of the scene be
    }
}

function ColorBasedOnXAxis(space) {
    const axis1_len = space.shape[1];
    space.iterate((d0, d1, d2) => {
        space.set_child_color({dim0: d0, dim1:d1, dim2:d2}, "hsl(" + parseInt(d1/axis1_len * 360) + ", 100%, 50%)")
    });
}

async function RunViz(Viz, SceneView) {
    let test_sequence = await Viz.CreateSequence("TestSeqeunce", 30);
    const initial_space = Viz.CreateIterationSpace({shape: [9, 18, 1],
                                            size: 1.0,
                                            padding: 0.4,
                                            position: {x: 0, y:0, z:0},
                                            color: 0x808080,
                                            rotation: GetDefault2dIterationPose()
    });



    const initial_space_label = Viz.CreateAxisLabel({
        space: initial_space,
        axis0: {color: 0x000000, arrow_thickness: 0.25, arrowhead_thickness: 0.5, arrow_length: 7, label: "i", label_pos: 2, label_size: 1.5},
        axis1: {color: 0x000000, arrow_thickness: 0.25, arrowhead_thickness: 0.5, arrow_length: 7, label: "j", label_pos: 2, label_size: 1.5},
    });

    const full_split_axis = initial_space.shape[1];
    const split_size = 6;
    ColorBasedOnXAxis(initial_space);

    const lighting = Viz.CreateIterationSpaceLighting({
        space: initial_space,
        show_lights: false,
        top_light: {color: 0xFFFFFF, panel_distance: 15, intensity : 0.85, distance : 30, decay : 2,  light_count: {x: 2, y: 2}, light_stride: {x: 12, y: 16}, offset: {x: 0, y: 0}},
        left_light: {color: 0xfeffd4, panel_distance: 10, intensity : 1, distance : 30, decay : 2,  light_count: {x: 2, y: 2}, light_stride: {x: 18, y: 18}, offset: {x: 0, y: 0}},
        right_light: {color: 0xc7f8ff, panel_distance: 10, intensity : 0.55, distance : 30, decay : 2.2,  light_count: {x: 2, y: 2}, light_stride: {x: 18, y: 18}, offset: {x: 0, y: 0}},
    });


    await Viz.SaveImage("PreSplit");
    await test_sequence.AddFrame(1000);

    // Remove initial iter space
    initial_space.remove();
    initial_space_label.remove();

    // Create 3 iteration spaces
    const split_space_start = -8.5;
    const split_space_padding = 8.4;
    const expand_factor = 0.5;
    let split_spaces = [];

    for(let i = 0; i < 3; ++i) {
        const space = Viz.CreateIterationSpace({shape: [9, split_size, 1],
                size: 1.0,
                padding: 0.4,
                position: {x: split_space_start + (i * (split_space_padding)), y:0, z:0},
                color: 0x808080,
                rotation: GetDefault2dIterationPose()
        });

        space.iterate((d0, d1, d2) => {
            space.set_child_color({dim0: d0, dim1: d1, dim2: d2}, "hsl(" + parseInt(((d1 + (split_size * i)) / full_split_axis) * 360) + ", 100%, 50%)")
        });

        split_spaces.push(space);
    }

    await Viz.SaveImage("Split");
    await test_sequence.AddFrame(1000);

    const duration = 2000;
    let target_pos = split_spaces[0].root_object.position.clone();
    target_pos.x -= 4.0;

    test_sequence.AddAnimation(TweenPosition(split_spaces[0].root_object, {target: target_pos.clone(), duration: duration}));

    target_pos = split_spaces[2].root_object.position.clone();
    target_pos.x += 4.0;
    test_sequence.AddAnimation(TweenPosition(split_spaces[2].root_object, {target: target_pos.clone(), duration: duration}));

    await test_sequence.RenderAllAnimations();

    // Relative rotation
    const target_angle = 1.57;
    for (let s = 0; s < split_spaces.length; ++s) {
        test_sequence.AddAnimation(
            TweenRotation(split_spaces[s].root_object,
            {
                target: {
                    y: "+" + target_angle
                },
                duration: duration
            })
        );
    }

    await test_sequence.RenderAllAnimations();
    await Viz.SaveImage("Rotated");

    target_pos = split_spaces[0].root_object.position.clone();
    target_pos.x += 10.5;

    test_sequence.AddAnimation(TweenPosition(split_spaces[0].root_object, {target: target_pos.clone(), duration: duration}));

    target_pos = split_spaces[2].root_object.position.clone();
    target_pos.x -= 10.5;
    test_sequence.AddAnimation(TweenPosition(split_spaces[2].root_object, {target: target_pos.clone(), duration: duration}));

    await test_sequence.RenderAllAnimations();
    await Viz.SaveImage("Contracted");

    // Delete the old spaces
    split_spaces.forEach((s) => s.remove());

    // Create a new iteration space
    let default_pose = GetDefault2dIterationPose();
    default_pose.y -= 1.57;

    const reshaped_space = Viz.CreateIterationSpace({shape: [9, split_size, 3],
            size: 1.0,
            padding: 0.4,
            position: {x: 0, y:0, z:0},
            color: 0x808080,
            rotation: default_pose
    });

    reshaped_space.iterate((d0, d1, d2) => {
        reshaped_space.set_child_color({dim0: d0, dim1: d1, dim2: d2}, "hsl(" + parseInt(((reshaped_space.shape[1] - 1 - d1) + (d2 * split_size))/full_split_axis * 360) + ", 100%, 50%)")
    });

    await Viz.SaveImage("ReshapedSpace");

    test_sequence.AddAnimation(
        TweenRotation(reshaped_space.root_object,
        {
            target: GetDefault3dIterationPose(),
            duration: duration
        })
    );

    await test_sequence.RenderAllAnimations();
    await Viz.SaveImage("FinalSpaceRotated");

    const reshaped_space_label = Viz.CreateAxisLabel({
        space: reshaped_space,
        axis0: {color: 0x000000, arrow_thickness: 0.25, arrowhead_thickness: 0.5, arrow_length: 7, label: "i", label_pos: 2, label_size: 1.5},
        axis1: {color: 0x000000, arrow_thickness: 0.25, arrowhead_thickness: 0.5, arrow_length: 7, label: "j", label_pos: 2, label_size: 1.5},
        axis2: {color: 0x000000, arrow_thickness: 0.25, arrowhead_thickness: 0.5, arrow_length: 7, label: "jj", label_pos: 2, label_size: 1.5},
    });

    await Viz.SaveImage("FinalSpaceRotatedLabel");
    test_sequence.AddFrame(1000);
    await test_sequence.FinishSequence();
}

