"""
Preferences management for the Depth Blur Filter application.
Handles loading and saving user preferences to preferences.ini file.
"""

import os
import configparser
from typing import Dict, Any


class PreferencesManager:
    """Manages application preferences with persistent storage."""
    
    def __init__(self, config_file: str = "preferences.ini"):
        """
        Initialize the preferences manager.
        
        Args:
            config_file: Path to the preferences file
        """
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.defaults = {
            'ui': {
                'dark_mode': 'false',
                'show_depth_map': 'false'
            },
            'processing': {
                'cpu_mode': 'false',
                'detail_level': 'high'
            },
            'depth': {
                'hole_filling_enabled': 'true',
                'background_correction_enabled': 'true',
                'smoothing_strength': 'medium'
            },
            'blur': {
                'blur_strength': '1',
                'focal_length': '50'
            }
        }
        self.load_preferences()
    
    def load_preferences(self) -> Dict[str, Any]:
        """
        Load preferences from the config file.
        Creates default file if it doesn't exist.
        
        Returns:
            Dictionary of loaded preferences
        """
        try:
            if os.path.exists(self.config_file):
                self.config.read(self.config_file)
                print(f"📁 Loaded preferences from {self.config_file}")
            else:
                print(f"📁 Preferences file not found, creating defaults")
                self._create_default_config()
            
            # Convert to dictionary format
            preferences = {}
            for section_name, section in self.config.items():
                if section_name != 'DEFAULT':
                    preferences[section_name] = dict(section)
            
            return preferences
            
        except Exception as e:
            print(f"❌ Error loading preferences: {e}")
            print("🔄 Using default preferences")
            self._create_default_config()
            return self._get_defaults_dict()
    
    def save_preferences(self, preferences: Dict[str, Any]) -> bool:
        """
        Save preferences to the config file.
        
        Args:
            preferences: Dictionary of preferences to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear existing config
            self.config.clear()
            
            # Add sections and values
            for section_name, section_data in preferences.items():
                self.config.add_section(section_name)
                for key, value in section_data.items():
                    self.config.set(section_name, key, str(value))
            
            # Write to file
            with open(self.config_file, 'w') as f:
                self.config.write(f)
            
            print(f"💾 Saved preferences to {self.config_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving preferences: {e}")
            return False
    
    def get_preference(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a specific preference value.
        
        Args:
            section: Section name (e.g., 'ui', 'processing')
            key: Key name (e.g., 'dark_mode', 'cpu_mode')
            default: Default value if not found
            
        Returns:
            Preference value or default
        """
        try:
            if self.config.has_section(section) and self.config.has_option(section, key):
                value = self.config.get(section, key)
                # Convert string values back to appropriate types
                return self._convert_value(value)
            else:
                return default
        except Exception as e:
            print(f"❌ Error getting preference {section}.{key}: {e}")
            return default
    
    def set_preference(self, section: str, key: str, value: Any) -> bool:
        """
        Set a specific preference value.
        
        Args:
            section: Section name
            key: Key name
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.config.has_section(section):
                self.config.add_section(section)
            
            self.config.set(section, key, str(value))
            return True
            
        except Exception as e:
            print(f"❌ Error setting preference {section}.{key}: {e}")
            return False
    
    def _create_default_config(self):
        """Create default configuration file."""
        try:
            for section_name, section_data in self.defaults.items():
                self.config.add_section(section_name)
                for key, value in section_data.items():
                    self.config.set(section_name, key, value)
            
            with open(self.config_file, 'w') as f:
                self.config.write(f)
            
            print(f"📝 Created default preferences file: {self.config_file}")
            
        except Exception as e:
            print(f"❌ Error creating default config: {e}")
    
    def _get_defaults_dict(self) -> Dict[str, Any]:
        """Get defaults as dictionary."""
        return self.defaults.copy()
    
    def _convert_value(self, value: str) -> Any:
        """
        Convert string value back to appropriate type.
        
        Args:
            value: String value from config file
            
        Returns:
            Converted value
        """
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def get_all_preferences(self) -> Dict[str, Any]:
        """
        Get all preferences as a dictionary.
        
        Returns:
            Dictionary of all preferences
        """
        preferences = {}
        for section_name in self.config.sections():
            preferences[section_name] = {}
            for key in self.config.options(section_name):
                value = self.config.get(section_name, key)
                preferences[section_name][key] = self._convert_value(value)
        
        return preferences
    
    def save_all_preferences(self, preferences: Dict[str, Any]) -> bool:
        """
        Save all preferences at once.
        
        Args:
            preferences: Dictionary of all preferences
            
        Returns:
            True if successful, False otherwise
        """
        return self.save_preferences(preferences)
