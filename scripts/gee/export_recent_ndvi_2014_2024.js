// Export annual growing-season Landsat NDVI for the Mongolian Plateau.
// Run in Google Earth Engine Code Editor.
//
// Purpose:
//   Regenerate valid 2014-2024 NDVI rasters because the local
//   Grassland_YYYY_NDVI.tif files under result/mask/2014-2024/maskresult
//   are all NaN. This script exports unmasked NDVI first; apply grassland
//   or subtype masks later during local sampling.

var WEST = 85;
var EAST = 125;
var SOUTH = 37;
var NORTH = 55;
var START_YEAR = 2014;
var END_YEAR = 2024;
var EXPORT_FOLDER = 'Mongolian_Plateau_recent_NDVI_2014_2024';
var SCALE = 30;

var roi = ee.Geometry.Rectangle([WEST, SOUTH, EAST, NORTH], null, false);

function maskLandsatC2L2(image) {
  var qa = image.select('QA_PIXEL');
  var fill = qa.bitwiseAnd(1 << 0).eq(0);
  var dilatedCloud = qa.bitwiseAnd(1 << 1).eq(0);
  var cirrus = qa.bitwiseAnd(1 << 2).eq(0);
  var cloud = qa.bitwiseAnd(1 << 3).eq(0);
  var cloudShadow = qa.bitwiseAnd(1 << 4).eq(0);
  var snow = qa.bitwiseAnd(1 << 5).eq(0);
  var mask = fill.and(dilatedCloud).and(cirrus).and(cloud).and(cloudShadow).and(snow);
  return image.updateMask(mask);
}

function addNdvi(image) {
  var red = image.select('SR_B4').multiply(0.0000275).add(-0.2);
  var nir = image.select('SR_B5').multiply(0.0000275).add(-0.2);
  var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
  return image.addBands(ndvi);
}

function annualNdvi(year) {
  var start = ee.Date.fromYMD(year, 5, 1);
  var end = ee.Date.fromYMD(year, 9, 30);
  var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterDate(start, end)
    .filterBounds(roi)
    .map(maskLandsatC2L2)
    .map(addNdvi);
  var l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterDate(start, end)
    .filterBounds(roi)
    .map(maskLandsatC2L2)
    .map(addNdvi);
  var collection = l8.merge(l9);
  return collection
    .select('NDVI')
    .median()
    .clip(roi)
    .toFloat()
    .set('year', year)
    .set('image_count', collection.size())
    .set('system:time_start', ee.Date.fromYMD(year, 7, 1).millis());
}

for (var year = START_YEAR; year <= END_YEAR; year++) {
  var ndvi = annualNdvi(year);
  print('NDVI image', year, 'image_count', ndvi.get('image_count'));
  Export.image.toDrive({
    image: ndvi,
    description: 'Recent_NDVI_' + year,
    folder: EXPORT_FOLDER,
    fileNamePrefix: 'Recent_NDVI_' + year,
    region: roi,
    scale: SCALE,
    crs: 'EPSG:4326',
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF',
    formatOptions: {
      cloudOptimized: true
    }
  });
}

Map.centerObject(roi, 5);
Map.addLayer(annualNdvi(2024), {
  min: 0,
  max: 0.8,
  palette: ['d73027', 'f46d43', 'fdae61', 'fee08b', 'd9ef8b', 'a6d96a', '66bd63', '1a9850']
}, 'Recent NDVI 2024');

