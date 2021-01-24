use std::ops::Sub;

pub fn line_unsigned(orig: &(u32, u32), end: &(u32, u32)) -> Vec<(u32, u32)> {
    let orig = (orig.0 as isize, orig.1 as isize);
    let end = (end.0 as isize, end.1 as isize);

    line(orig, end)
        .into_iter()
        .filter(|(x, y)| x > &0 && y > &0)
        .map(|(x, y)| (x as u32, y as u32))
        .collect()
}

pub fn line(orig: (isize, isize), end: (isize, isize)) -> Vec<(isize, isize)> {
    if (end.1 - orig.1).abs() < (end.0 - orig.0).abs() {
        if orig.0 > end.0 {
            line_low(end, orig)
        } else {
            line_low(orig, end)
        }
    } else {
        if orig.1 > end.1 {
            line_high(end, orig)
        } else {
            line_high(orig, end)
        }
    }
}

fn line_low(orig: (isize, isize), end: (isize, isize)) -> Vec<(isize, isize)> {
    // compute the points to color for a shallow line
    let mut res = Vec::new();
    let mut del = (end.0 - orig.0, end.1 - orig.1);
    let mut y_inc = 1;
    if del.1 < 0 {
        y_inc = -1;
        del.1 = -del.1;
    }
    let mut d = (2 * del.1) - del.0;
    let mut y = orig.1;
    for x in (orig.0)..=(end.0) {
        res.push((x, y));
        if d > 0 {
            y += y_inc;
            d += 2 * (del.1 - del.0);
        } else {
            d += 2 * del.1;
        }
    }
    res
}

fn line_high(orig: (isize, isize), end: (isize, isize)) -> Vec<(isize, isize)> {
    // compute the points to color for a shallow line
    let mut res = Vec::new();
    let mut del = (end.0 - orig.0, end.1 - orig.1);
    let mut x_inc = 1;
    if del.0 < 0 {
        x_inc = -1;
        del.0 = -del.0;
    }
    let mut d = (2 * del.0) - del.1;
    let mut x = orig.0;
    for y in (orig.1)..=(end.1) {
        res.push((x, y));
        if d > 0 {
            x += x_inc;
            d += 2 * (del.0 - del.1);
        } else {
            d += 2 * del.0;
        }
    }
    res
}
