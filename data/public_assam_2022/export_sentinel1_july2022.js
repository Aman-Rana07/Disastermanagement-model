// Export calibrated, terrain-corrected Sentinel-1 VV/VH backscatter
// in dB for comparison with the NRSC Cachar flood map issued 17 July 2022.
//
// Run in the Google Earth Engine Code Editor:
// https://code.earthengine.google.com/
// After starting the Drive export task, download the GeoTIFF to:
// data/public_assam_2022/sentinel1/

var region = ee.Geometry.Rectangle(
  [92.40942, 24.30928, 93.27984, 25.19587],
  'EPSG:4326',
  false
);

var collection = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(region)
  .filterDate('2022-07-10', '2022-07-21')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains(
    'transmitterReceiverPolarisation', 'VV'
  ))
  .filter(ee.Filter.listContains(
    'transmitterReceiverPolarisation', 'VH'
  ))
  .select(['VV', 'VH']);

print('Matching Sentinel-1 scenes, 10-20 July 2022:', collection.size());
print(
  'Scene acquisition times:',
  collection.aggregate_array('system:time_start')
    .map(function(time) {
      return ee.Date(time).format('YYYY-MM-dd HH:mm');
    })
);

var vvVh = collection.median().clip(region);
var validObservations = collection.select('VV').count().clip(region);

Map.centerObject(region, 9);
Map.addLayer(vvVh.select('VV'), {
  min: -25,
  max: 0
}, 'VV median (dB), 10-20 July 2022');
Map.addLayer(validObservations, {
  min: 0,
  max: 3,
  palette: ['red', 'yellow', 'green']
}, 'Number of valid VV observations');
Map.addLayer(region, {color: 'yellow'}, 'NRSC Cachar map extent');

Export.image.toDrive({
  image: vvVh.unmask(-9999),
  description: 'Assam_Cachar_S1_VV_VH_20220710_20220720',
  folder: 'AssamFlood2022',
  fileNamePrefix: 'assam_cachar_s1_vv_vh_20220710_20220720',
  region: region,
  crs: 'EPSG:32646',
  scale: 10,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true,
    noData: -9999
  }
});
