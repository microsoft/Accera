////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
////////////////////////////////////////////////////////////////////////////////////////////////////

function GetCommonConstants() {
    return {
        name: "Slides",
        pixel_width: 1000,
        pixel_height: 800,
        background_color: 0xFFFFFFFF,
        ambient_light_color: 0xFFFFFF,
        ambient_light_intensity: 0.3,
        shadow_map_resolution: 2048,
        arrow_color: 0x333333,
        line_color: 0x333333,
        ball_color: 0xFFFFFF,
        highlight_color: 0x05EE20,
        padding_color1: 0x4444AA,
        padding_color2: 0xCC2222,
        fuse_color2: 0xcc8bf0,
        fuse_color1: 0xffffff,//ff6666,
        padding_color: 0x444444,
    };
}

// Args:
//    h in [0, 360]
//    s, v in [0, 1]
// Returns:
//    r, g, b in [0, 1]
function hsvToRgb(h, s, v) {
    let f = (n, k = (n + h / 60) % 6) => v - v * s * Math.max(Math.min(k, 4 - k, 1), 0);
    return [f(5), f(3), f(1)];
}

function GenerateColors(count, min_saturation = 0.2, min_value = 0.5) {
    const i = 360 / (count - 1);
    var r = [];
    for (let x = 0; x < count; ++x) {
        hue = i * x
        saturation = Math.max(min_saturation, Math.random())
        value = Math.max(min_value, Math.random())
        rgb = hsvToRgb(hue, saturation, value);
        hex = (rgb[2] * 255) | ((rgb[1] * 255) << 8) | ((rgb[0] * 255) << 16);
        r.push(hex);
    }
    return r;
}

function GetKnownColors(count, shuffle) {
    // Alternative to the color generation code above
    // this selects from known colors, up to a maximum
    // Source: https://www.html-code-generator.com/html/color-codes-and-names
    const colors = [
        0xF0F8FF,
        0xFAEBD7,
        0x00FFFF,
        0x7FFFD4,
        0xF0FFFF,
        0xF5F5DC,
        0xFFE4C4,
        0x000000,
        0xFFEBCD,
        0x0000FF,
        0x8A2BE2,
        0xA52A2A,
        0xDEB887,
        0x5F9EA0,
        0x7FFF00,
        0xD2691E,
        0xFF7F50,
        0x6495ED,
        0xFFF8DC,
        0xDC143C,
        0x00FFFF,
        0x00008B,
        0x008B8B,
        0xB8860B,
        0xA9A9A9,
        0xA9A9A9,
        0x006400,
        0xBDB76B,
        0x8B008B,
        0x556B2F,
        0xFF8C00,
        0x9932CC,
        0x8B0000,
        0xE9967A,
        0x8FBC8F,
        0x483D8B,
        0x2F4F4F,
        0x2F4F4F,
        0x00CED1,
        0x9400D3,
        0xFF1493,
        0x00BFFF,
        0x696969,
        0x696969,
        0x1E90FF,
        0xB22222,
        0xFFFAF0,
        0x228B22,
        0xFF00FF,
        0xDCDCDC,
        0xF8F8FF,
        0xFFD700,
        0xDAA520,
        0x808080,
        0x808080,
        0x008000,
        0xADFF2F,
        0xF0FFF0,
        0xFF69B4,
        0xCD5C5C,
        0x4B0082,
        0xFFFFF0,
        0xF0E68C,
        0xE6E6FA,
        0xFFF0F5,
        0x7CFC00,
        0xFFFACD,
        0xADD8E6,
        0xF08080,
        0xE0FFFF,
        0xFAFAD2,
        0xD3D3D3,
        0xD3D3D3,
        0x90EE90,
        0xFFB6C1,
        0xFFA07A,
        0x20B2AA,
        0x87CEFA,
        0x778899,
        0x778899,
        0xB0C4DE,
        0xFFFFE0,
        0x00FF00,
        0x32CD32,
        0xFAF0E6,
        0xFF00FF,
        0x800000,
        0x66CDAA,
        0x0000CD,
        0xBA55D3,
        0x9370DB,
        0x3CB371,
        0x7B68EE,
        0x00FA9A,
        0x48D1CC,
        0xC71585,
        0x191970,
        0xF5FFFA,
        0xFFE4E1,
        0xFFE4B5,
        0xFFDEAD,
        0x000080,
        0xFDF5E6,
        0x808000,
        0x6B8E23,
        0xFFA500,
        0xFF4500,
        0xDA70D6,
        0xEEE8AA,
        0x98FB98,
        0xAFEEEE,
        0xDB7093,
        0xFFEFD5,
        0xFFDAB9,
        0xCD853F,
        0xFFC0CB,
        0xDDA0DD,
        0xB0E0E6,
        0x800080,
        0x663399,
        0xFF0000,
        0xBC8F8F,
        0x4169E1,
        0x8B4513,
        0xFA8072,
        0xF4A460,
        0x2E8B57,
        0xFFF5EE,
        0xA0522D,
        0xC0C0C0,
        0x87CEEB,
        0x6A5ACD,
        0x708090,
        0x708090,
        0xFFFAFA,
        0x00FF7F,
        0x4682B4,
        0xD2B48C,
        0x008080,
        0xD8BFD8,
        0xFF6347,
        0x40E0D0,
        0xEE82EE,
        0xF5DEB3,
        0xFFFFFF,
        0xF5F5F5,
        0xFFFF00,
        0x9ACD32,
    ];
    if (count > colors.length) {
        throw "count is too large! Maximum is " + colors.length;
    }

    if (shuffle) {
        return colors.slice(0, count).sort(() => Math.random() - 0.5);
    }
    return colors.slice(0, count);
}


