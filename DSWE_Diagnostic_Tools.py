# ============================================================================
# DIAGNOSTIC CODE: Analyze 8-Class DSWE Distribution
# ============================================================================
# Complete version with all dependencies included

import ee
import geopandas as gpd
import numpy as np

# ============================================================================
# HELPER FUNCTIONS (Required Dependencies)
# ============================================================================

def load_roi(shapefile_path):
    """Load ROI from a shapefile and return as an EE Geometry."""
    gdf = gpd.read_file(shapefile_path)
    return ee.Geometry.Polygon(gdf.unary_union.__geo_interface__["coordinates"])

def Mndwi(image):
    """Calculate MNDWI."""
    return image.normalizedDifference(['Green', 'Swir1']).rename('mndwi')

def Mbsrv(image):
    """Calculate MBSRV."""
    return image.select(['Green']).add(image.select(['Red'])).rename('mbsrv')

def Mbsrn(image):
    """Calculate MBSRN."""
    return image.select(['Nir']).add(image.select(['Swir1'])).rename('mbsrn')

def Ndvi(image):
    """Calculate NDVI."""
    return image.normalizedDifference(['Nir', 'Red']).rename('ndvi')

def Awesh(image):
    """Calculate AWEsh."""
    return image.expression(
        'Blue + 2.5 * Green + (-1.5) * mbsrn + (-0.25) * Swir2',
        {
            'Blue': image.select(['Blue']),
            'Green': image.select(['Green']),
            'mbsrn': Mbsrn(image).select(['mbsrn']),
            'Swir2': image.select(['Swir2'])
        }
    ).rename('awesh')


# ============================================================================
# MAIN DIAGNOSTIC FUNCTIONS
# ============================================================================

def analyze_class_distribution(image_path, roi_path=None, sample_scale=30):
    """
    Analyze the distribution of classes in an 8-class DSWE image.
    
    Parameters:
    -----------
    image_path : str
        Path to GEE asset (e.g., 'projects/ee-okavango/.../Water_Class_2013_07')
    roi_path : str, optional
        Path to ROI shapefile or GEE asset. If None, uses image footprint
    sample_scale : int
        Scale for analysis (default 30m for Landsat)
        
    Returns:
    --------
    dict : Statistics for each class including pixel counts and areas
    """
    
    # Load the classification image
    classification = ee.Image(image_path)
    
    # Load ROI if provided, otherwise use image geometry
    if roi_path:
        if roi_path.startswith('projects/'):
            # GEE asset
            roi = ee.FeatureCollection(roi_path).geometry()
        else:
            # Local shapefile
            roi = load_roi(roi_path)
    else:
        roi = classification.geometry()
    
    # Get class counts
    class_stats = {}
    
    print("=" * 70)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"Image: {image_path}")
    print(f"Scale: {sample_scale}m")
    print("-" * 70)
    
    # Analyze each class
    for class_val in range(8):
        # Create mask for this class
        class_mask = classification.eq(class_val)
        
        # Calculate pixel count
        pixel_count = class_mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=sample_scale,
            maxPixels=1e13
        ).getInfo()['water_class']
        
        # Calculate area (km²)
        area_image = class_mask.multiply(ee.Image.pixelArea().divide(1e6))
        area_km2 = area_image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=sample_scale,
            maxPixels=1e13
        ).getInfo()['water_class']
        
        class_stats[class_val] = {
            'pixels': int(pixel_count),
            'area_km2': round(area_km2, 2)
        }
        
        # Print results
        class_names = {
            0: "Dry/No Water",
            1: "Low Confidence Water",
            2: "Vegetated Flooded - Low Conf",
            3: "Moderate Confidence Water",
            4: "Vegetated Flooded - Mod Conf",
            5: "High Confidence Open Water",
            6: "High Confidence Vegetated Water",
            7: "Very High Confidence Open Water"
        }
        
        status = "⚠️ EMPTY" if pixel_count == 0 else "✓"
        pct = (pixel_count / sum([class_stats[i]['pixels'] for i in class_stats])) * 100 if sum([class_stats[i]['pixels'] for i in class_stats]) > 0 else 0
        
        print(f"Class {class_val} {status}: {class_names[class_val]}")
        print(f"  Pixels: {int(pixel_count):,} ({pct:.1f}%)")
        print(f"  Area: {area_km2:,.2f} km²")
        print()
    
    # Calculate total water area
    water_mask = classification.gt(0)
    total_water_area = water_mask.multiply(
        ee.Image.pixelArea().divide(1e6)
    ).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=sample_scale,
        maxPixels=1e13
    ).getInfo()['water_class']
    
    print("-" * 70)
    print(f"TOTAL WATER AREA: {total_water_area:,.2f} km²")
    print("=" * 70)
    
    return class_stats


def analyze_test_combinations(image_path, composite_path, roi_path, sample_points=1000):
    """
    Analyze which test combinations are occurring and which tests are passing.
    This helps diagnose why certain classes have no pixels.
    
    Parameters:
    -----------
    image_path : str
        Path to classification GEE asset
    composite_path : str
        Path to the source Landsat composite
    roi_path : str
        Path to ROI
    sample_points : int
        Number of random points to sample
        
    Returns:
    --------
    dict : Statistics on test passage rates
    """
    
    print("=" * 70)
    print("TEST COMBINATION ANALYSIS")
    print("=" * 70)
    
    # Load images
    classification = ee.Image(image_path)
    composite = ee.Image(composite_path)
    
    if roi_path.startswith('projects/'):
        roi = ee.FeatureCollection(roi_path).geometry()
    else:
        roi = load_roi(roi_path)
    
    # Get dynamic thresholds from classification metadata
    swir1_threshold = classification.get('swir1_threshold')
    swir2_threshold = classification.get('swir2_threshold')
    swir1_threshold_val = float(swir1_threshold.getInfo())
    swir2_threshold_val = float(swir2_threshold.getInfo())

    print(f"SWIR1 Threshold: {swir1_threshold_val:.4f}")
    print(f"SWIR2 Threshold: {swir2_threshold_val:.4f}")
    print()
    
    # Calculate all indices
    mndwi = Mndwi(composite)
    mbsrv = Mbsrv(composite)
    mbsrn = Mbsrn(composite)
    awesh = Awesh(composite)
    ndvi = Ndvi(composite)
    
    # Recreate tests
    t1 = mndwi.gt(0.124)
    t2 = mbsrv.gt(mbsrn)
    t3 = awesh.gt(0)
    
    t4 = (mndwi.gt(0.0)
          .And(composite.select('Swir1').lt(swir1_threshold_val))
          .And(composite.select('Swir2').lt(swir2_threshold_val))
          .And(ndvi.lt(0.3))
          .And(composite.select('Nir').lt(0.15)))

    t5 = (mndwi.gt(-0.1)
          .And(composite.select('Swir2').lt(swir2_threshold_val))
          .And(ndvi.gt(0.3))
          .And(ndvi.lt(0.75))
          .And(composite.select('Nir').lt(0.30)))

    t6 = (mndwi.gt(-0.2)
          .And(composite.select('Swir1').lt(swir1_threshold_val * 1.1))
          .And(composite.select('Swir2').lt(swir2_threshold_val * 1.1))
          .And(ndvi.gt(0.4))
          .And(ndvi.lt(0.85))
          .And(composite.select('Nir').lt(0.35)))

    t7 = (composite.select('Swir2').lt(swir2_threshold_val * 0.85)
          .And(composite.select('Blue').lt(0.15))
          .And(composite.select('Nir').lt(0.40)))
    
    # Stack all tests
    test_stack = ee.Image.cat([
        t1.rename('t1'),
        t2.rename('t2'),
        t3.rename('t3'),
        t4.rename('t4'),
        t5.rename('t5'),
        t6.rename('t6'),
        t7.rename('t7'),
        classification.rename('class')
    ])
    
    # Generate random sample points
    sample = test_stack.sample(
        region=roi,
        scale=30,
        numPixels=sample_points,
        seed=42
    )
    
    # Convert to Python
    sample_list = sample.getInfo()['features']
    
    # Analyze test passage rates
    test_counts = {f't{i}': 0 for i in range(1, 8)}
    class_counts = {i: 0 for i in range(8)}
    
    for feature in sample_list:
        props = feature['properties']
        class_val = props['class']
        class_counts[class_val] = class_counts.get(class_val, 0) + 1
        
        for i in range(1, 8):
            if props.get(f't{i}', 0) == 1:
                test_counts[f't{i}'] += 1
    
    print("TEST PASSAGE RATES (% of sampled pixels):")
    print("-" * 70)
    test_names = {
        't1': 'Strong MNDWI (> 0.124)',
        't2': 'MBSRV > MBSRN',
        't3': 'AWEsh > 0',
        't4': 'Open Water (dynamic SWIR, NDVI < 0.3)',
        't5': 'Veg Water Moderate (NDVI 0.3-0.75)',
        't6': 'Veg Water Aggressive (NDVI 0.4-0.85)',
        't7': 'Moisture Detection (SWIR2 < 0.85*threshold)'
    }
    
    for test in ['t1', 't2', 't3', 't4', 't5', 't6', 't7']:
        rate = (test_counts[test] / len(sample_list)) * 100
        status = "✓" if rate > 10 else "⚠️ LOW" if rate > 0 else "❌ NONE"
        print(f"{test.upper()} {status}: {test_names[test]}")
        print(f"  Pass Rate: {rate:.1f}%")
        print()
    
    print("-" * 70)
    print("CLASS DISTRIBUTION (% of sampled pixels):")
    print("-" * 70)
    for class_val in range(8):
        count = class_counts.get(class_val, 0)
        pct = (count / len(sample_list)) * 100
        status = "⚠️ DOMINANT" if pct > 50 else "✓"
        print(f"Class {class_val} {status}: {pct:.1f}% ({count} pixels)")
    
    print("=" * 70)
    
    return {
        'test_counts': test_counts,
        'class_counts': class_counts,
        'total_samples': len(sample_list)
    }


def diagnose_class_1_dominance(image_path, composite_path, roi_path):
    """
    Comprehensive diagnosis for why Class 1 dominates.
    Runs all diagnostic tests in sequence.
    """
    
    print("\n" + "=" * 70)
    print("COMPREHENSIVE DIAGNOSIS: CLASS 1 DOMINANCE")
    print("=" * 70 + "\n")
    
    # 1. Class distribution
    print("STEP 1: Class Distribution")
    print("-" * 70)
    stats = analyze_class_distribution(image_path, roi_path)
    
    # 2. Test passage rates
    print("\n\nSTEP 2: Test Combination Analysis")
    print("-" * 70)
    test_stats = analyze_test_combinations(image_path, composite_path, roi_path, sample_points=5000)
    
    # 3. Check thresholds
    print("\n\nSTEP 3: Threshold Values")
    print("-" * 70)
    classification = ee.Image(image_path)
    swir1_thresh = classification.get('swir1_threshold').getInfo()
    swir2_thresh = classification.get('swir2_threshold').getInfo()
    swir1_wet = classification.get('swir1_wet_mode').getInfo()
    swir1_dry = classification.get('swir1_dry_mode').getInfo()
    swir2_wet = classification.get('swir2_wet_mode').getInfo()
    swir2_dry = classification.get('swir2_dry_mode').getInfo()
    
    print(f"SWIR1 Threshold: {swir1_thresh:.4f}")
    print(f"  Wet Mode: {swir1_wet:.4f}")
    print(f"  Dry Mode: {swir1_dry:.4f}")
    if swir1_dry != swir1_wet:
        trough_pos = ((swir1_thresh - swir1_wet) / (swir1_dry - swir1_wet) * 100)
        print(f"  Trough Position: {trough_pos:.1f}% between modes")
    print()
    print(f"SWIR2 Threshold: {swir2_thresh:.4f}")
    print(f"  Wet Mode: {swir2_wet:.4f}")
    print(f"  Dry Mode: {swir2_dry:.4f}")
    if swir2_dry != swir2_wet:
        trough_pos = ((swir2_thresh - swir2_wet) / (swir2_dry - swir2_wet) * 100)
        print(f"  Trough Position: {trough_pos:.1f}% between modes")
    
    # 4. Diagnose likely issues
    print("\n\nSTEP 4: Diagnosis")
    print("-" * 70)
    
    issues = []
    fixes = []
    
    # Check if high confidence tests are passing
    if test_stats['test_counts']['t4'] < test_stats['total_samples'] * 0.05:
        issues.append("⚠️ Test 4 (Open Water) barely passing - SWIR thresholds likely too strict")
        fixes.append("Relax SWIR constraints and/or NDVI threshold in test 4")
    
    if test_stats['test_counts']['t5'] < test_stats['total_samples'] * 0.05:
        issues.append("⚠️ Test 5 (Veg Water Moderate) barely passing - check NDVI ranges")
        fixes.append("Lower NDVI minimum in test 5 from 0.3 to 0.20")
    
    if test_stats['test_counts']['t6'] < test_stats['total_samples'] * 0.05:
        issues.append("⚠️ Test 6 (Veg Water Aggressive) barely passing - check NDVI ranges")
        fixes.append("Lower NDVI minimum in test 6 from 0.4 to 0.25")
    
    if swir1_thresh > 0.25:
        issues.append(f"⚠️ SWIR1 threshold ({swir1_thresh:.3f}) higher than typical")
        fixes.append("Lower min_swir1 constraint to 0.10 in calculate_dynamic_thresholds()")
    
    if swir2_thresh > 0.20:
        issues.append(f"⚠️ SWIR2 threshold ({swir2_thresh:.3f}) higher than typical")
        fixes.append("Lower min_swir2 constraint to 0.08 in calculate_dynamic_thresholds()")
    
    if len(issues) == 0:
        print("✓ No obvious threshold/test issues detected")
        print()
        print("Possible causes:")
        print("  1. Hierarchical filtering may be too aggressive")
        print("  2. Class combination logic may need adjustment")
        print("  3. Most water genuinely is low confidence (ephemeral/shallow)")
    else:
        print("IDENTIFIED ISSUES:")
        for issue in issues:
            print(f"  {issue}")
        print()
        print("RECOMMENDED FIXES:")
        for i, fix in enumerate(fixes, 1):
            print(f"  {i}. {fix}")
    
    print("\n" + "=" * 70 + "\n")
    
    return {
        'stats': stats,
        'test_stats': test_stats,
        'thresholds': {
            'swir1': swir1_thresh,
            'swir2': swir2_thresh,
            'swir1_wet': swir1_wet,
            'swir1_dry': swir1_dry,
            'swir2_wet': swir2_wet,
            'swir2_dry': swir2_dry
        },
        'issues': issues,
        'fixes': fixes
    }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
from DSWE_Diagnostic_Tools import diagnose_class_1_dominance

results = diagnose_class_1_dominance(
    image_path='projects/ee-okavango/assets/water_masks/Adapted_DSWE_LS_30m_v1/Water_Class/Water_Class_2017_07',
    composite_path='projects/ee-okavango/assets/water_masks/monthly_DSWE_Landsat_30m_v2/Source_LS_Composites/Composite_2017_07',
    roi_path=r"C:\path\to\shapefile.shp"
)
"""