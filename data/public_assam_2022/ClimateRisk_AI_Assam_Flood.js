// =======================================
// ClimateRisk-AI
// Assam Flood Mapping V4
// =======================================

// Assam AOI
var assam = ee.Geometry.Rectangle(
    [91.0, 26.0, 92.5, 27.5]
);

Map.centerObject(assam, 8);

// =======================================
// Sentinel-1
// =======================================

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(assam)
    .filter(ee.Filter.listContains(
        'transmitterReceiverPolarisation',
        'VV'
    ))
    .filter(ee.Filter.eq(
        'instrumentMode',
        'IW'
    ))
    .select('VV');

// =======================================
// Before Flood
// =======================================

var before = s1
    .filterDate(
        '2022-05-01',
        '2022-05-31'
    )
    .median()
    .focal_mean(50, 'circle', 'meters')
    .clip(assam);

// =======================================
// During Flood
// =======================================

var during = s1
    .filterDate(
        '2022-06-15',
        '2022-06-30'
    )
    .median()
    .focal_mean(50, 'circle', 'meters')
    .clip(assam);

// =======================================
// dB -> Linear
// =======================================

var beforeLinear =
    ee.Image(10).pow(
        before.divide(10)
    );

var duringLinear =
    ee.Image(10).pow(
        during.divide(10)
    );

// =======================================
// Ratio
// =======================================

var ratio =
    duringLinear.divide(
        beforeLinear
    );

// =======================================
// Flood Candidates
// =======================================

var flood =
    ratio.lt(0.70);

// =======================================
// Remove Permanent Water
// =======================================

var water = ee.Image(
    'JRC/GSW1_4/GlobalSurfaceWater'
);

var seasonality =
    water.select('seasonality');

var permanentWater =
    seasonality.gte(10);

flood = flood.and(
    permanentWater.not()
);

// =======================================
// SRTM DEM
// =======================================

var dem = ee.Image('USGS/SRTMGL1_003');

var slope = ee.Terrain.slope(dem);

// Remove steep areas
flood = flood.and(
    slope.lt(5)
);
// =======================================
// Remove Small Isolated Patches
// =======================================

var floodClean = flood.updateMask(
    flood.connectedPixelCount(
        100,
        true
    ).gte(50)
);

// =======================================
// Flood Area
// =======================================

// Flood area in square meters
var floodAreaImage =
    floodClean.selfMask()
    .multiply(ee.Image.pixelArea());

// Sum flooded area
var floodArea =
    floodAreaImage.reduceRegion({
        reducer: ee.Reducer.sum(),
        geometry: assam,
        scale: 30,
        maxPixels: 1e13
    });

// Get first available value
var floodKm2 =
    ee.Number(
        floodArea.values().get(0)
    ).divide(1e6);

print(
    'Flood Area (km²)',
    floodKm2
);

// =======================================
// Display
// =======================================

Map.addLayer(
    floodClean.selfMask(),
    {
        palette:['cyan']
    },
    'Flood Extent'
);

Map.addLayer(
    ee.Image().paint(
        assam,
        1,
        2
    ),
    {
        palette:['yellow']
    },
    'Boundary'
);

print(
    'Before Images',
    s1.filterDate(
        '2022-05-01',
        '2022-05-31'
    ).size()
);

print(
    'Flood Images',
    s1.filterDate(
        '2022-06-15',
        '2022-06-30'
    ).size()
);

// =======================================
// Export Flood Raster
// =======================================

Export.image.toDrive({
  image: floodClean,
  description: 'Assam_Flood_Extent_2022',
  folder: 'ClimateRisk_AI',
  fileNamePrefix: 'assam_flood_extent_2022',
  region: assam,
  scale: 30,
  maxPixels: 1e13
});

// =======================================
// Raster to Polygon
// =======================================

var floodVector = floodClean.selfMask()
  .reduceToVectors({
    geometry: assam,
    scale: 30,
    geometryType: 'polygon',
    eightConnected: true,
    maxPixels: 1e13
  });

Map.addLayer(
  floodVector,
  {color: 'red'},
  'Flood Polygons'
);

// =======================================
// Export GeoJSON
// =======================================

Export.table.toDrive({
  collection: floodVector,
  description: 'Assam_Flood_Polygons_2022',
  folder: 'ClimateRisk_AI',
  fileFormat: 'GeoJSON'
});
