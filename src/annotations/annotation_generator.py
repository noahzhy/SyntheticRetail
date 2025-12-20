"""
Annotation generation from rendered images and instance masks
Supports Pascal VOC XML format
"""
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from xml.dom import minidom


@dataclass
class BoundingBox:
    """Bounding box representation"""
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    
    @property
    def width(self) -> int:
        return self.xmax - self.xmin
    
    @property
    def height(self) -> int:
        return self.ymax - self.ymin
    
    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class Annotation:
    """Single object annotation"""
    sku_id: str
    category: str
    bbox: BoundingBox
    polygon: List[Tuple[int, int]]
    occlusion_ratio: float
    truncated: bool = False
    difficult: bool = False


class AnnotationGenerator:
    """Generate annotations from instance masks"""
    
    def __init__(
        self,
        min_bbox_area: int = 100,
        polygon_tolerance: float = 2.0,
        occlusion_threshold: float = 0.3
    ):
        self.min_bbox_area = min_bbox_area
        self.polygon_tolerance = polygon_tolerance
        self.occlusion_threshold = occlusion_threshold
    
    def generate_annotations(
        self,
        instance_mask: np.ndarray,
        depth_map: np.ndarray,
        sku_mapping: Dict[int, str],
        category_mapping: Dict[str, str]
    ) -> List[Annotation]:
        """Generate annotations from instance mask and depth map"""
        annotations = []
        
        # Get unique instance IDs
        instance_ids = np.unique(instance_mask)
        instance_ids = instance_ids[instance_ids > 0]  # Skip background
        
        for instance_id in instance_ids:
            # Get SKU for this instance
            sku_id = sku_mapping.get(instance_id, f"SKU{instance_id:03d}")
            category = category_mapping.get(sku_id, "unknown")
            
            # Create binary mask for this instance
            binary_mask = (instance_mask == instance_id).astype(np.uint8)
            
            # Calculate bounding box
            bbox = self._calculate_bbox(binary_mask)
            
            if bbox is None or bbox.area < self.min_bbox_area:
                continue
            
            # Calculate polygon contour
            polygon = self._calculate_polygon(binary_mask)
            
            # Calculate occlusion ratio
            occlusion_ratio = self._calculate_occlusion(
                binary_mask, depth_map, instance_mask, instance_id
            )
            
            # Check if truncated (touching image border)
            truncated = self._is_truncated(binary_mask)
            
            # Create annotation
            annotation = Annotation(
                sku_id=sku_id,
                category=category,
                bbox=bbox,
                polygon=polygon,
                occlusion_ratio=occlusion_ratio,
                truncated=truncated,
                difficult=False
            )
            
            annotations.append(annotation)
        
        return annotations
    
    def _calculate_bbox(self, mask: np.ndarray) -> Optional[BoundingBox]:
        """Calculate bounding box from binary mask"""
        coords = np.column_stack(np.where(mask > 0))
        
        if len(coords) == 0:
            return None
        
        ymin, xmin = coords.min(axis=0)
        ymax, xmax = coords.max(axis=0)
        
        return BoundingBox(
            xmin=int(xmin),
            ymin=int(ymin),
            xmax=int(xmax),
            ymax=int(ymax)
        )
    
    def _calculate_polygon(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Calculate simplified polygon contour"""
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return []
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Simplify polygon
        epsilon = self.polygon_tolerance
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to list of tuples
        polygon = [(int(point[0][0]), int(point[0][1])) for point in approx]
        
        return polygon
    
    def _calculate_occlusion(
        self,
        binary_mask: np.ndarray,
        depth_map: np.ndarray,
        instance_mask: np.ndarray,
        instance_id: int
    ) -> float:
        """Calculate occlusion ratio using depth information"""
        # For 2D rendering, we use a simplified approach
        # Real occlusion would require checking for overlapping instances
        
        # Get bounding box
        bbox = self._calculate_bbox(binary_mask)
        if bbox is None:
            return 0.0
        
        # Count pixels of this instance within its bbox
        roi_mask = instance_mask[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax]
        visible_pixels = np.sum(roi_mask == instance_id)
        
        # Count pixels of OTHER instances within this bbox (occlusion)
        other_instance_pixels = np.sum((roi_mask > 0) & (roi_mask != instance_id))
        
        total_pixels = visible_pixels + other_instance_pixels
        
        if total_pixels == 0:
            return 0.0
        
        # Occlusion ratio: other instances / total foreground
        occlusion_ratio = other_instance_pixels / total_pixels
        
        return float(np.clip(occlusion_ratio, 0.0, 1.0))
    
    def _is_truncated(self, mask: np.ndarray) -> bool:
        """Check if object touches image border"""
        h, w = mask.shape
        
        # Check if any pixels on border are part of the mask
        if np.any(mask[0, :]) or np.any(mask[-1, :]):
            return True
        if np.any(mask[:, 0]) or np.any(mask[:, -1]):
            return True
        
        return False
    
    def export_pascal_voc(
        self,
        annotations: List[Annotation],
        image_path: str,
        image_size: Tuple[int, int, int],
        output_path: str
    ):
        """Export annotations in Pascal VOC XML format"""
        width, height, depth = image_size
        
        # Create XML structure
        annotation_xml = ET.Element('annotation')
        
        # Add folder
        folder = ET.SubElement(annotation_xml, 'folder')
        folder.text = 'images'
        
        # Add filename
        filename = ET.SubElement(annotation_xml, 'filename')
        filename.text = Path(image_path).name
        
        # Add path
        path = ET.SubElement(annotation_xml, 'path')
        path.text = str(image_path)
        
        # Add source
        source = ET.SubElement(annotation_xml, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'SyntheticRetail'
        
        # Add size
        size = ET.SubElement(annotation_xml, 'size')
        width_elem = ET.SubElement(size, 'width')
        width_elem.text = str(width)
        height_elem = ET.SubElement(size, 'height')
        height_elem.text = str(height)
        depth_elem = ET.SubElement(size, 'depth')
        depth_elem.text = str(depth)
        
        # Add segmented
        segmented = ET.SubElement(annotation_xml, 'segmented')
        segmented.text = '1'
        
        # Add objects
        for ann in annotations:
            obj = ET.SubElement(annotation_xml, 'object')
            
            name = ET.SubElement(obj, 'name')
            name.text = ann.category
            
            pose = ET.SubElement(obj, 'pose')
            pose.text = 'Unspecified'
            
            truncated = ET.SubElement(obj, 'truncated')
            truncated.text = '1' if ann.truncated else '0'
            
            difficult = ET.SubElement(obj, 'difficult')
            difficult.text = '1' if ann.difficult else '0'
            
            # Add custom fields
            sku = ET.SubElement(obj, 'sku_id')
            sku.text = ann.sku_id
            
            occlusion = ET.SubElement(obj, 'occlusion')
            occlusion.text = f"{ann.occlusion_ratio:.3f}"
            
            # Add bounding box
            bndbox = ET.SubElement(obj, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            xmin.text = str(ann.bbox.xmin)
            ymin = ET.SubElement(bndbox, 'ymin')
            ymin.text = str(ann.bbox.ymin)
            xmax = ET.SubElement(bndbox, 'xmax')
            xmax.text = str(ann.bbox.xmax)
            ymax = ET.SubElement(bndbox, 'ymax')
            ymax.text = str(ann.bbox.ymax)
            
            # Add polygon
            if ann.polygon:
                polygon = ET.SubElement(obj, 'polygon')
                for i, (x, y) in enumerate(ann.polygon):
                    point = ET.SubElement(polygon, f'point{i}')
                    x_elem = ET.SubElement(point, 'x')
                    x_elem.text = str(x)
                    y_elem = ET.SubElement(point, 'y')
                    y_elem.text = str(y)
        
        # Pretty print and save
        xml_str = minidom.parseString(ET.tostring(annotation_xml)).toprettyxml(indent="  ")
        
        with open(output_path, 'w') as f:
            f.write(xml_str)
