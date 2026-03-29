"""
Traffic Sign Occlusion Recovery Pipeline
Using Classification Model Only (No YOLO)

This module can be imported and used in other Python files.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional


class TrafficSignInpainter:
    """
    Traffic sign inpainting pipeline that uses classification and template-based recovery.
    
    Example usage:
        from traffic_sign_inpainter import TrafficSignInpainter
        
        inpainter = TrafficSignInpainter(
            classification_model_path='model.h5',
            class_names_path='classes.npy',
            templates_dir='templates/'
        )
        
        result, stats = inpainter.process_image('sign.jpg')
    """
    
    def __init__(
        self,
        classification_model_path: str,
        class_names_path: str,
        templates_dir: str = 'data/templates',
        inpaint_method: str = 'hybrid',
        conservative_mask: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the inpainting pipeline
        
        Args:
            classification_model_path: Path to Keras classification model (.h5)
            class_names_path: Path to class names numpy file (.npy)
            templates_dir: Directory containing clean template images
            inpaint_method: Method to use ('template', 'opencv', or 'hybrid')
            conservative_mask: If True, only detect obvious occlusions
            verbose: If True, print progress messages
        """
        self.verbose = verbose
        self.classification_model = None
        self.class_names = None
        self.templates = {}
        self.inpaint_method = inpaint_method
        self.conservative_mask = conservative_mask
        
        # Load classification model
        self._load_classification_model(classification_model_path, class_names_path)
        
        # Load templates
        self.templates = self._load_templates(templates_dir)
        
        # Stats tracking
        self.stats = {
            'total_processed': 0,
            'successful_inpaints': 0,
            'failed_inpaints': 0
        }
        
        if self.verbose:
            mode = "CONSERVATIVE" if conservative_mask else "AGGRESSIVE"
            print(f"ℹ Using {mode} mask detection")
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    def _load_classification_model(self, model_path: str, class_names_path: str):
        """Load the Keras classification model and class names"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Classification model not found: {model_path}")
        
        if not Path(class_names_path).exists():
            raise FileNotFoundError(f"Class names file not found: {class_names_path}")
        
        try:
            from tensorflow import keras
            self.classification_model = keras.models.load_model(model_path)
            self.class_names = np.load(class_names_path, allow_pickle=True)
            
            self._log(f"✓ Classification model loaded successfully")
            self._log(f"✓ Class names loaded: {len(self.class_names)} classes")
            
        except Exception as e:
            raise RuntimeError(f"Could not load classification model: {e}")
    
    def _load_templates(self, templates_dir: str) -> Dict[str, np.ndarray]:
        """Load clean template images for each sign class from subfolders"""
        templates = {}
        templates_path = Path(templates_dir)
        
        if not templates_path.exists():
            self._log(f"⚠ Warning: Templates directory not found: {templates_dir}")
            return templates
        
        # Iterate through subdirectories (each represents a class)
        for class_dir in templates_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            
            # Load all images from this class directory
            image_files = list(class_dir.glob('*.png')) + \
                          list(class_dir.glob('*.jpg')) + \
                          list(class_dir.glob('*.jpeg'))
            
            if not image_files:
                continue
            
            # Use the first image as the template
            template_path = image_files[0]
            template = cv2.imread(str(template_path))
            
            if template is not None:
                templates[class_name] = template
                self._log(f"✓ Loaded template: {class_name}")
        
        self._log(f"\n✓ Total templates loaded: {len(templates)}\n")
        return templates
    
    def classify_sign(self, roi: np.ndarray, img_size: int = 64) -> Tuple[str, float]:
        """
        Use classification model to determine sign class
        
        Args:
            roi: Cropped region of interest
            img_size: Input size for classifier (default: 64)
            
        Returns:
            (predicted_class, confidence)
        """
        try:
            # Preprocess
            resized = cv2.resize(roi, (img_size, img_size))
            normalized = resized / 255.0
            input_img = np.expand_dims(normalized, axis=0)
            
            # Make prediction
            predictions = self.classification_model.predict(input_img, verbose=0)
            confidence = float(np.max(predictions))
            predicted_class_idx = int(np.argmax(predictions))
            
            # Get class name
            if predicted_class_idx < len(self.class_names):
                predicted_class = self.class_names[predicted_class_idx]
            else:
                predicted_class = 'unknown'
            
            return predicted_class, confidence
        
        except Exception as e:
            self._log(f"⚠ Classification error: {e}")
            return 'unknown', 0.0
    
    def generate_occlusion_mask(self, image: np.ndarray, conservative: bool = None) -> np.ndarray:
        """
        Generate binary mask for occluded regions
        
        Args:
            image: Traffic sign ROI
            conservative: If True, only detect obvious occlusions. If None, uses instance setting.
            
        Returns:
            Binary mask (255 = occluded/missing, 0 = clear/visible)
        """
        if conservative is None:
            conservative = self.conservative_mask
        
        if conservative:
            # CONSERVATIVE MODE: Only detect very obvious occlusions
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Only very dark regions
            _, dark_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
            
            # Only very black regions
            black_mask = cv2.inRange(gray, 0, 30)
            
            # Combine
            combined_mask = cv2.bitwise_or(dark_mask, black_mask)
            
            # Clean up
            kernel = np.ones((3, 3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            return combined_mask
        
        else:
            # AGGRESSIVE MODE: Detect all potential occlusions
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
            
            # Edge detection
            edges = cv2.Canny(gray, 30, 120)
            kernel = np.ones((7, 7), np.uint8)
            edge_mask = cv2.dilate(edges, kernel, iterations=2)
            edge_mask = 255 - edge_mask
            
            # Color consistency check
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            low_saturation = hsv[:, :, 1] < 80
            low_value = hsv[:, :, 2] < 80
            color_mask = ((low_saturation | low_value) * 255).astype(np.uint8)
            
            # Very dark/black regions
            black_mask = cv2.inRange(gray, 0, 40)
            
            # Combine all masks
            combined_mask = cv2.bitwise_or(dark_mask, color_mask)
            combined_mask = cv2.bitwise_or(combined_mask, edge_mask)
            combined_mask = cv2.bitwise_or(combined_mask, black_mask)
            
            # Morphological operations
            kernel_small = np.ones((3, 3), np.uint8)
            kernel_large = np.ones((5, 5), np.uint8)
            
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
            combined_mask = cv2.dilate(combined_mask, kernel_small, iterations=1)
            
            return combined_mask
    
    def inpaint_opencv(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Perform OpenCV-based inpainting
        
        Args:
            image: ROI to inpaint
            mask: Binary mask of occluded regions
            
        Returns:
            Inpainted image
        """
        inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        return inpainted
    
    def inpaint_template(
        self, 
        image: np.ndarray, 
        sign_class: str,
        mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Template-based inpainting using clean reference images
        ONLY replaces occluded regions marked in mask
        
        Args:
            image: Traffic sign ROI
            sign_class: Predicted class of the sign
            mask: Binary mask (255 = occluded, 0 = keep original)
            
        Returns:
            Inpainted image (original + template only in occluded areas)
        """
        if sign_class not in self.templates:
            self._log(f"⚠ Warning: No template for class '{sign_class}'")
            return image
        
        h, w = image.shape[:2]
        
        # Get template and resize
        template = self.templates[sign_class]
        template_resized = cv2.resize(template, (w, h))
        
        if mask is not None:
            # ONLY replace pixels where mask is white (occluded)
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            blended = (image * (1 - mask_3ch) + template_resized * mask_3ch).astype(np.uint8)
            return blended
        else:
            self._log("⚠ Warning: No mask provided, returning original image")
            return image
    
    def inpaint_hybrid(
        self,
        image: np.ndarray,
        sign_class: str,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Hybrid approach: OpenCV for background, template for sign structure
        
        Args:
            image: Traffic sign ROI
            sign_class: Predicted class
            mask: Occlusion mask
            
        Returns:
            Inpainted image
        """
        # First, use OpenCV inpainting for texture
        inpainted = self.inpaint_opencv(image, mask)
        
        # Then, blend with template for structure
        if sign_class in self.templates:
            h, w = image.shape[:2]
            template = cv2.resize(self.templates[sign_class], (w, h))
            
            # Use mask to determine blending weight
            mask_ratio = np.sum(mask > 0) / mask.size
            template_weight = min(0.6, mask_ratio)
            
            inpainted = cv2.addWeighted(
                inpainted, 1-template_weight,
                template, template_weight, 0
            )
        
        return inpainted
    
    def process_image(
        self, 
        image_path: str,
        bbox: Optional[List[int]] = None,
        output_path: str = None,
        visualize: bool = True,
        manual_mask: Optional[np.ndarray] = None,
        mask_threshold: float = 0.05,
        save_components: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        Main pipeline: classify -> inpaint
        
        Args:
            image_path: Input image path (can be cropped sign or full image)
            bbox: Optional [x1, y1, x2, y2] if working with full image
            output_path: Where to save result (optional)
            visualize: Whether to create visualization
            manual_mask: Optional pre-computed mask (if None, auto-generate)
            mask_threshold: Skip inpainting if occlusion ratio below this (default: 0.05 = 5%)
            save_components: Whether to save individual components (mask, original, etc.)
            
        Returns:
            (processed_image, stats_dict)
        """
        self._log(f"\n{'='*60}")
        self._log(f"Processing: {image_path}")
        self._log(f"{'='*60}\n")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Extract ROI if bbox provided
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            roi = image[y1:y2, x1:x2].copy()
            self._log(f"Using bbox: {bbox}")
        else:
            roi = image.copy()
            self._log("Processing entire image as traffic sign")
        
        original_roi = roi.copy()
        
        # Step 1: Classify the sign
        self._log("\nStep 1: Classifying sign...")
        sign_class, confidence = self.classify_sign(roi)
        self._log(f"  - Predicted class: {sign_class}")
        self._log(f"  - Confidence: {confidence:.2%}")
        
        if confidence < 0.3:
            self._log(f"⚠ Warning: Low confidence ({confidence:.2%})")
        
        # Step 2: Generate occlusion mask
        self._log("\nStep 2: Generating occlusion mask...")
        
        if manual_mask is not None:
            self._log("  - Using manual mask provided")
            mask = manual_mask
        else:
            self._log("  - Generating automatic mask")
            mask = self.generate_occlusion_mask(roi)
        
        occlusion_ratio = np.sum(mask > 0) / mask.size
        self._log(f"  - Occlusion ratio: {occlusion_ratio*100:.1f}%")
        
        if occlusion_ratio > 0.7:
            self._log(f"  ⚠ WARNING: Mask detecting {occlusion_ratio*100:.1f}% as occluded!")
        
        # Skip inpainting if occlusion is below threshold
        skip_inpainting = occlusion_ratio < mask_threshold
        
        if skip_inpainting:
            self._log(f"  ℹ Occlusion below threshold, skipping inpainting")
            result_roi = roi.copy()
        else:
            # Step 3: Perform inpainting
            self._log(f"\nStep 3: Inpainting using '{self.inpaint_method}' method...")
            
            if self.inpaint_method == 'template':
                result_roi = self.inpaint_template(roi, sign_class, mask)
            elif self.inpaint_method == 'opencv':
                result_roi = self.inpaint_opencv(roi, mask)
            elif self.inpaint_method == 'hybrid':
                result_roi = self.inpaint_hybrid(roi, sign_class, mask)
            else:
                self._log(f"⚠ Unknown method '{self.inpaint_method}', using hybrid")
                result_roi = self.inpaint_hybrid(roi, sign_class, mask)
            
            # Update stats
            self.stats['total_processed'] += 1
            if sign_class in self.templates:
                self.stats['successful_inpaints'] += 1
            else:
                self.stats['failed_inpaints'] += 1
        
        # Step 4: Re-classify to check improvement
        self._log("\nStep 4: Re-classifying inpainted sign...")
        new_class, new_confidence = self.classify_sign(result_roi)
        self._log(f"  - New class: {new_class}")
        self._log(f"  - New confidence: {new_confidence:.2%}")
        
        confidence_improvement = new_confidence - confidence
        self._log(f"  - Confidence change: {confidence_improvement:+.2%}")
        
        # Create result image
        if bbox is not None:
            result_image = image.copy()
            result_image[y1:y2, x1:x2] = result_roi
        else:
            result_image = result_roi
        
        self._log(f"\n{'='*60}")
        self._log("RESULTS:")
        self._log(f"{'='*60}")
        self._log(f"Original: {sign_class} ({confidence:.2%})")
        self._log(f"Final: {new_class} ({new_confidence:.2%})")
        self._log(f"Improvement: {confidence_improvement:+.2%}")
        self._log(f"{'='*60}\n")
        
        # Create visualization
        comparison = None
        if visualize:
            comparison = self._create_visualization(
                original_roi, mask, result_roi, sign_class, 
                new_class, occlusion_ratio
            )
        
        # Save results
        if output_path:
            self._save_results(
                output_path, result_image, comparison, 
                original_roi, mask, result_roi, save_components
            )
        
        return result_image, {
            'original_class': sign_class,
            'original_confidence': confidence,
            'final_class': new_class,
            'final_confidence': new_confidence,
            'confidence_improvement': confidence_improvement,
            'occlusion_ratio': occlusion_ratio,
            'mask': mask,
            'comparison': comparison
        }
    
    def _create_visualization(
        self, 
        original_roi: np.ndarray,
        mask: np.ndarray,
        result_roi: np.ndarray,
        sign_class: str,
        new_class: str,
        occlusion_ratio: float
    ) -> np.ndarray:
        """Create comparison grid visualization"""
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        def add_label(img, text, color=(255, 255, 255)):
            img_copy = img.copy()
            cv2.putText(img_copy, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            return img_copy
        
        labeled_original = add_label(original_roi, f"Original: {sign_class[:15]}")
        labeled_mask = add_label(mask_colored, f"Mask: {occlusion_ratio*100:.0f}%")
        labeled_result = add_label(result_roi, f"Result: {new_class[:15]}")
        
        row1 = np.hstack([labeled_original, labeled_mask])
        
        if sign_class in self.templates:
            template_resized = cv2.resize(
                self.templates[sign_class], 
                (result_roi.shape[1], result_roi.shape[0])
            )
            labeled_template = add_label(template_resized, "Template")
            row2 = np.hstack([labeled_result, labeled_template])
        else:
            black = np.zeros_like(result_roi)
            labeled_black = add_label(black, "No Template", (100, 100, 100))
            row2 = np.hstack([labeled_result, labeled_black])
        
        comparison = np.vstack([row1, row2])
        return comparison
    
    def _save_results(
        self,
        output_path: str,
        result_image: np.ndarray,
        comparison: Optional[np.ndarray],
        original_roi: np.ndarray,
        mask: np.ndarray,
        result_roi: np.ndarray,
        save_components: bool
    ):
        """Save all output files"""
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main result
        cv2.imwrite(output_path, result_image)
        self._log(f"✓ Saved result to: {output_path}")
        
        # Save comparison
        if comparison is not None:
            comparison_path = str(output_path).replace('.', '_comparison.')
            cv2.imwrite(comparison_path, comparison)
            self._log(f"✓ Saved comparison to: {comparison_path}")
        
        # Save individual components
        if save_components:
            components_dir = output_path_obj.parent / 'components'
            components_dir.mkdir(exist_ok=True)
            
            stem = output_path_obj.stem
            cv2.imwrite(str(components_dir / f'{stem}_original.jpg'), original_roi)
            cv2.imwrite(str(components_dir / f'{stem}_mask.jpg'), mask)
            cv2.imwrite(str(components_dir / f'{stem}_result.jpg'), result_roi)
            self._log(f"✓ Saved components to: {components_dir}")
    
    def batch_process(
        self,
        input_dir: str,
        output_dir: str,
        save_stats: bool = True
    ) -> List[Dict]:
        """
        Process multiple pre-cropped traffic sign images
        
        Args:
            input_dir: Directory containing cropped sign images
            output_dir: Directory to save results
            save_stats: Whether to save statistics JSON
            
        Returns:
            List of statistics dictionaries for each processed image
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = list(input_path.glob('*.jpg')) + \
                      list(input_path.glob('*.png')) + \
                      list(input_path.glob('*.jpeg'))
        
        all_stats = []
        
        self._log(f"\n{'='*60}")
        self._log(f"BATCH PROCESSING: {len(image_files)} images")
        self._log(f"{'='*60}\n")
        
        for idx, img_file in enumerate(image_files, 1):
            self._log(f"\n[{idx}/{len(image_files)}] Processing {img_file.name}...")
            output_file = output_path / f"processed_{img_file.name}"
            
            try:
                _, stats = self.process_image(
                    str(img_file),
                    bbox=None,
                    output_path=str(output_file),
                    visualize=True
                )
                
                all_stats.append({
                    'filename': img_file.name,
                    **{k: v for k, v in stats.items() if k not in ['mask', 'comparison']}
                })
                
            except Exception as e:
                self._log(f"✗ Error processing {img_file.name}: {str(e)}")
                continue
        
        # Calculate overall statistics
        if all_stats:
            avg_conf_before = np.mean([s['original_confidence'] for s in all_stats])
            avg_conf_after = np.mean([s['final_confidence'] for s in all_stats])
            avg_improvement = np.mean([s['confidence_improvement'] for s in all_stats])
            avg_occlusion = np.mean([s['occlusion_ratio'] for s in all_stats])
            
            self._log(f"\n{'='*60}")
            self._log("BATCH PROCESSING COMPLETE")
            self._log(f"{'='*60}")
            self._log(f"Images processed: {len(all_stats)}")
            self._log(f"Average confidence before: {avg_conf_before:.2%}")
            self._log(f"Average confidence after: {avg_conf_after:.2%}")
            self._log(f"Average improvement: {avg_improvement:+.2%}")
            self._log(f"Average occlusion: {avg_occlusion*100:.1f}%")
            self._log(f"{'='*60}\n")
            
            if save_stats:
                stats_file = output_path / 'processing_stats.json'
                with open(stats_file, 'w') as f:
                    json.dump({
                        'summary': {
                            'total_images': len(all_stats),
                            'avg_confidence_before': float(avg_conf_before),
                            'avg_confidence_after': float(avg_conf_after),
                            'avg_improvement': float(avg_improvement),
                            'avg_occlusion': float(avg_occlusion)
                        },
                        'per_image': all_stats
                    }, f, indent=2)
                self._log(f"✓ Saved statistics to: {stats_file}")
        
        return all_stats


# Convenience function for quick usage
def process_single_sign(
    image_path: str,
    model_path: str,
    class_names_path: str,
    templates_dir: str,
    output_path: str = None,
    bbox: Optional[List[int]] = None,
    method: str = 'hybrid',
    conservative: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to process a single sign without creating an instance
    
    Args:
        image_path: Path to input image
        model_path: Path to classification model
        class_names_path: Path to class names file
        templates_dir: Path to templates directory
        output_path: Optional path to save result
        bbox: Optional bounding box [x1, y1, x2, y2]
        method: Inpainting method ('template', 'opencv', or 'hybrid')
        conservative: Use conservative mask detection
        
    Returns:
        (processed_image, stats_dict)
    """
    inpainter = TrafficSignInpainter(
        classification_model_path=model_path,
        class_names_path=class_names_path,
        templates_dir=templates_dir,
        inpaint_method=method,
        conservative_mask=conservative,
        verbose=True
    )
    
    return inpainter.process_image(
        image_path=image_path,
        bbox=bbox,
        output_path=output_path,
        visualize=True
    )


if __name__ == "__main__":
    # Example usage when run directly
    import sys
    
    print("This module is designed to be imported.")
    print("\nExample usage in your main application:")
    print("-" * 60)
    print("""
from traffic_sign_inpainter import TrafficSignInpainter

# Create inpainter instance
inpainter = TrafficSignInpainter(
    classification_model_path='path/to/model.h5',
    class_names_path='path/to/classes.npy',
    templates_dir='path/to/templates/',
    inpaint_method='hybrid',
    conservative_mask=True,
    verbose=True  # Set to False to suppress output
)

# Process single image
result, stats = inpainter.process_image(
    image_path='test_sign.jpg',
    output_path='output/result.jpg'
)

print(f"Original: {stats['original_class']} ({stats['original_confidence']:.2%})")
print(f"Final: {stats['final_class']} ({stats['final_confidence']:.2%})")

# Or use the convenience function
from traffic_sign_inpainter import process_single_sign

result, stats = process_single_sign(
    image_path='test_sign.jpg',
    model_path='model.h5',
    class_names_path='classes.npy',
    templates_dir='templates/',
    output_path='output/result.jpg'
)
    """)
    print("-" * 60)