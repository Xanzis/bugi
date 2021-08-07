use image::Rgb;

pub const STDCOL: [Rgb<u8>; 4] = [
    Rgb([255, 0, 0]),
    Rgb([255, 255, 0]),
    Rgb([0, 0, 255]),
    Rgb([255, 0, 255]),
];

pub const BACK: Rgb<u8> = Rgb([10, 10, 10]);

pub fn hot_map(val: f64) -> Rgb<u8> {
    let val = val.min(1.0).max(0.0);
    let r = (val * 3.0 * 255.0).clamp(0.0, 255.0);
    let g = ((val * 3.0 - 1.0) * 255.0).clamp(0.0, 255.0);
    let b = ((val * 3.0 - 2.0) * 255.0).clamp(0.0, 255.0);

    Rgb([r.round() as u8, g.round() as u8, b.round() as u8])
}

pub fn hot_map_boxed() -> Box<dyn Fn(f64) -> Rgb<u8>> {
    Box::new(|x| hot_map(x))
}

fn peak(top: (f64, f64), slope: f64, val: f64) -> f64 {
    if val < top.0 {
        slope * (val - top.0) + top.1
    } else {
        -1.0 * slope * (val - top.0) + top.1
    }
}

pub fn rgb_map(val: f64) -> Rgb<u8> {
    let r = (peak((0.75, 1.5), 4.0, val) * 255.0).clamp(0.0, 255.0);
    let g = (peak((0.50, 1.5), 4.0, val) * 255.0).clamp(0.0, 255.0);
    let b = (peak((0.25, 1.5), 4.0, val) * 255.0).clamp(0.0, 255.0);

    Rgb([r.round() as u8, g.round() as u8, b.round() as u8])
}

pub fn rgb_map_boxed() -> Box<dyn Fn(f64) -> Rgb<u8>> {
    Box::new(|x| rgb_map(x))
}
