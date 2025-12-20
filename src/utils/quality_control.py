"""
Quality control and validation for generated data
"""
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

from src.annotations.annotation_generator import Annotation


@dataclass
class QCReport:
    """Quality control report for a single image"""
    scene_id: str
    passed: bool
    num_products: int
    num_valid_annotations: int
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class QualityController:
    """Quality control and validation"""
    
    def __init__(
        self,
        min_products: int = 5,
        max_occlusion_rate: float = 0.8,
        check_bbox_validity: bool = True,
        check_empty_images: bool = True
    ):
        self.min_products = min_products
        self.max_occlusion_rate = max_occlusion_rate
        self.check_bbox_validity = check_bbox_validity
        self.check_empty_images = check_empty_images
    
    def validate_scene(
        self,
        scene_id: str,
        annotations: List[Annotation],
        image_size: Tuple[int, int]
    ) -> QCReport:
        """Validate a generated scene"""
        issues = []
        metrics = {}
        
        width, height = image_size
        
        # Check number of products
        num_products = len(annotations)
        metrics['num_products'] = num_products
        
        if self.check_empty_images and num_products == 0:
            issues.append("Empty image: no products detected")
        
        if num_products < self.min_products:
            issues.append(f"Too few products: {num_products} < {self.min_products}")
        
        # Check individual annotations
        valid_annotations = 0
        high_occlusion_count = 0
        out_of_bounds_count = 0
        
        for ann in annotations:
            is_valid = True
            
            # Check occlusion
            if ann.occlusion_ratio > self.max_occlusion_rate:
                high_occlusion_count += 1
                is_valid = False
            
            # Check bbox validity
            if self.check_bbox_validity:
                if ann.bbox.xmin < 0 or ann.bbox.ymin < 0:
                    out_of_bounds_count += 1
                    is_valid = False
                if ann.bbox.xmax > width or ann.bbox.ymax > height:
                    out_of_bounds_count += 1
                    is_valid = False
                if ann.bbox.width <= 0 or ann.bbox.height <= 0:
                    out_of_bounds_count += 1
                    is_valid = False
            
            if is_valid:
                valid_annotations += 1
        
        metrics['valid_annotations'] = valid_annotations
        metrics['high_occlusion_count'] = high_occlusion_count
        metrics['out_of_bounds_count'] = out_of_bounds_count
        
        if high_occlusion_count > 0:
            issues.append(f"{high_occlusion_count} products with high occlusion (>{self.max_occlusion_rate})")
        
        if out_of_bounds_count > 0:
            issues.append(f"{out_of_bounds_count} products with invalid bounding boxes")
        
        # Calculate average occlusion
        if annotations:
            avg_occlusion = np.mean([ann.occlusion_ratio for ann in annotations])
            metrics['avg_occlusion'] = float(avg_occlusion)
        else:
            metrics['avg_occlusion'] = 0.0
        
        # Determine if scene passed
        passed = len(issues) == 0 and valid_annotations >= self.min_products
        
        return QCReport(
            scene_id=scene_id,
            passed=passed,
            num_products=num_products,
            num_valid_annotations=valid_annotations,
            issues=issues,
            metrics=metrics
        )
    
    def should_resample(self, report: QCReport) -> bool:
        """Determine if scene should be regenerated"""
        return not report.passed


@dataclass
class DistributionStats:
    """Distribution statistics"""
    total_scenes: int
    total_products: int
    sku_distribution: Dict[str, int]
    category_distribution: Dict[str, int]
    occlusion_distribution: Dict[str, int]  # bins of occlusion ranges
    failed_scenes: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_scenes': self.total_scenes,
            'total_products': self.total_products,
            'sku_distribution': self.sku_distribution,
            'category_distribution': self.category_distribution,
            'occlusion_distribution': self.occlusion_distribution,
            'failed_scenes': self.failed_scenes
        }


class DistributionMonitor:
    """Monitor and track data distribution"""
    
    def __init__(self):
        self.sku_counts: Dict[str, int] = {}
        self.category_counts: Dict[str, int] = {}
        self.occlusion_bins: Dict[str, int] = {
            '0.0-0.2': 0,
            '0.2-0.4': 0,
            '0.4-0.6': 0,
            '0.6-0.8': 0,
            '0.8-1.0': 0
        }
        self.total_scenes = 0
        self.total_products = 0
        self.failed_scenes = 0
    
    def update(self, annotations: List[Annotation], passed: bool = True):
        """Update distribution statistics"""
        self.total_scenes += 1
        
        if not passed:
            self.failed_scenes += 1
            return
        
        for ann in annotations:
            self.total_products += 1
            
            # Update SKU counts
            self.sku_counts[ann.sku_id] = self.sku_counts.get(ann.sku_id, 0) + 1
            
            # Update category counts
            self.category_counts[ann.category] = self.category_counts.get(ann.category, 0) + 1
            
            # Update occlusion distribution
            occ = ann.occlusion_ratio
            if occ < 0.2:
                self.occlusion_bins['0.0-0.2'] += 1
            elif occ < 0.4:
                self.occlusion_bins['0.2-0.4'] += 1
            elif occ < 0.6:
                self.occlusion_bins['0.4-0.6'] += 1
            elif occ < 0.8:
                self.occlusion_bins['0.6-0.8'] += 1
            else:
                self.occlusion_bins['0.8-1.0'] += 1
    
    def get_stats(self) -> DistributionStats:
        """Get current distribution statistics"""
        return DistributionStats(
            total_scenes=self.total_scenes,
            total_products=self.total_products,
            sku_distribution=dict(self.sku_counts),
            category_distribution=dict(self.category_counts),
            occlusion_distribution=dict(self.occlusion_bins),
            failed_scenes=self.failed_scenes
        )
    
    def generate_report(self) -> str:
        """Generate human-readable report"""
        stats = self.get_stats()
        
        report = []
        report.append("=" * 60)
        report.append("DISTRIBUTION REPORT")
        report.append("=" * 60)
        report.append(f"Total Scenes: {stats.total_scenes}")
        report.append(f"Failed Scenes: {stats.failed_scenes}")
        report.append(f"Total Products: {stats.total_products}")
        report.append("")
        
        report.append("SKU Distribution:")
        for sku, count in sorted(stats.sku_distribution.items()):
            percentage = (count / stats.total_products * 100) if stats.total_products > 0 else 0
            report.append(f"  {sku}: {count} ({percentage:.1f}%)")
        report.append("")
        
        report.append("Category Distribution:")
        for category, count in sorted(stats.category_distribution.items()):
            percentage = (count / stats.total_products * 100) if stats.total_products > 0 else 0
            report.append(f"  {category}: {count} ({percentage:.1f}%)")
        report.append("")
        
        report.append("Occlusion Distribution:")
        for bin_range, count in sorted(stats.occlusion_distribution.items()):
            percentage = (count / stats.total_products * 100) if stats.total_products > 0 else 0
            report.append(f"  {bin_range}: {count} ({percentage:.1f}%)")
        report.append("=" * 60)
        
        return "\n".join(report)
