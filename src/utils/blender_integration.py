"""
Blender integration wrapper for the synthetic data pipeline
Detects Blender installation and manages rendering process
"""
import subprocess
import shutil
import os
from pathlib import Path
from typing import Optional, Tuple
import json


class BlenderIntegration:
    """Manages Blender installation and rendering"""
    
    def __init__(self, blender_path: Optional[str] = None):
        """
        Initialize Blender integration
        
        Args:
            blender_path: Path to Blender executable. If None, will search in PATH
        """
        # If a path is provided, verify it exists
        if blender_path and blender_path != "blender":
            if os.path.exists(blender_path):
                self.blender_path = blender_path
            else:
                print(f"Specified Blender path does not exist: {blender_path}")
                self.blender_path = self._find_blender()
        else:
            # Search for Blender
            self.blender_path = self._find_blender()
        
        self.available = self.blender_path is not None
        
        if self.available:
            print(f"Blender found at: {self.blender_path}")
        else:
            print("Blender not found. Please install Blender or specify path.")
    
    def _find_blender(self) -> Optional[str]:
        """Find Blender executable in system PATH"""
        # Try common Blender executable names
        blender_names = ['blender', 'blender.exe', 'Blender']
        
        for name in blender_names:
            path = shutil.which(name)
            if path and os.path.exists(path):
                return path
        
        # Try common installation locations
        common_paths = [
            '/usr/bin/blender',
            '/usr/local/bin/blender',
            'C:\\Program Files\\Blender Foundation\\Blender 3.6\\blender.exe',
            'C:\\Program Files\\Blender Foundation\\Blender 3.5\\blender.exe',
            'C:\\Program Files\\Blender Foundation\\Blender\\blender.exe',
            '/Applications/Blender.app/Contents/MacOS/Blender',
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def get_version(self) -> Optional[str]:
        """Get Blender version"""
        if not self.available:
            return None
        
        try:
            result = subprocess.run(
                [self.blender_path, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # First line usually contains version
                return result.stdout.split('\n')[0]
        except Exception as e:
            print(f"Error getting Blender version: {e}")
        
        return None
    
    def render_scene(
        self,
        recipe_path: str,
        output_dir: str,
        renderer_script: str
    ) -> Tuple[bool, str]:
        """
        Render a scene using Blender
        
        Args:
            recipe_path: Path to scene recipe JSON
            output_dir: Output directory for rendered images
            renderer_script: Path to Blender Python script
            
        Returns:
            Tuple of (success, message)
        """
        if not self.available:
            return False, "Blender not available"
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Build Blender command
        cmd = [
            self.blender_path,
            '--background',  # Run without UI
            '--python', renderer_script,  # Run Python script
            '--',  # Separator for script arguments
            '--recipe', recipe_path,
            '--output', output_dir
        ]
        
        try:
            # Run Blender
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                return True, "Rendering successful"
            else:
                error_msg = f"Blender failed with code {result.returncode}"
                if result.stderr:
                    error_msg += f"\n{result.stderr[-500:]}"  # Last 500 chars of error
                return False, error_msg
                
        except subprocess.TimeoutExpired:
            return False, "Rendering timeout (>5 minutes)"
        except Exception as e:
            return False, f"Rendering error: {str(e)}"
    
    def install_instructions(self) -> str:
        """Get installation instructions for current platform"""
        import platform
        
        system = platform.system()
        
        instructions = {
            'Linux': """
Install Blender on Linux:

Ubuntu/Debian:
    sudo apt update
    sudo apt install blender

Or download from: https://www.blender.org/download/

After installation, run: python main.py --blender-path /path/to/blender
""",
            'Darwin': """
Install Blender on macOS:

1. Download from: https://www.blender.org/download/
2. Install to /Applications
3. Run: python main.py --blender-path /Applications/Blender.app/Contents/MacOS/Blender
""",
            'Windows': """
Install Blender on Windows:

1. Download from: https://www.blender.org/download/
2. Install (default: C:\\Program Files\\Blender Foundation\\Blender)
3. Run: python main.py --blender-path "C:\\Program Files\\Blender Foundation\\Blender 3.6\\blender.exe"
"""
        }
        
        return instructions.get(system, "Download Blender from: https://www.blender.org/download/")
