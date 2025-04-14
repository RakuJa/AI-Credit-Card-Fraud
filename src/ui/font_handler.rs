use egui::FontData;
use egui::FontDefinitions;
use egui::FontFamily;
use std::collections::BTreeMap;
macro_rules! load_font {
    ($ctx: expr, $($name:literal, $path:literal),*) => {

        let mut fonts = FontDefinitions::default();

        $(
            fonts.font_data.insert($name.into(), FontData::from_static(include_bytes!($path)).into());
            let mut newfam = BTreeMap::new();

            newfam.insert(FontFamily::Name($name.into()), vec![$name.to_owned()]);
            fonts.families.append(&mut newfam);
        )*

        $ctx.set_fonts(fonts);
    };
}

pub fn load_fonts(ctx: &egui::Context) {
    load_font!(
        ctx,
        "Pixelify",
        "../../ui/fonts/PixelifySans-VariableFont_wght.ttf"
    );
    /*
    load_font!(
        ctx,
        "GoodTimesRg",
        "../../ui/fonts/Good-times-rg.ttf"
    );

     */
}
