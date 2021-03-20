use image::Rgb;

pub const STDCOL: [Rgb<u8>; 4] = [
    Rgb([255, 0, 0]),
    Rgb([255, 255, 0]),
    Rgb([0, 0, 255]),
    Rgb([255, 0, 255]),
];

pub fn hot_map(val: f64) -> Rgb<u8> {
    let val = val.min(1.0).max(0.0);
    let r = (val * 3.0).min(1.0) * 255.0;
    let g = (val * 3.0 - 1.0).min(1.0).max(0.0) * 255.0;
    let b = (val * 3.0 - 2.0).max(0.0) * 255.0;

    Rgb([r.round() as u8, g.round() as u8, b.round() as u8])
}
