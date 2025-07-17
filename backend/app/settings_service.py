"""Settings management service"""
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from app.models import UserSettings


class SettingsService:
    """Service for managing user settings"""
    
    @staticmethod
    def get_setting(db: Session, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a setting value by key"""
        setting = db.query(UserSettings).filter(UserSettings.setting_key == key).first()
        return setting.setting_value if setting else default
    
    @staticmethod
    def set_setting(db: Session, key: str, value: str, description: str = "") -> UserSettings:
        """Set a setting value"""
        setting = db.query(UserSettings).filter(UserSettings.setting_key == key).first()
        
        if setting:
            setting.setting_value = value
            setting.description = description
        else:
            setting = UserSettings(
                setting_key=key,
                setting_value=value,
                description=description
            )
            db.add(setting)
        
        db.commit()
        db.refresh(setting)
        return setting
    
    @staticmethod
    def get_all_settings(db: Session) -> Dict[str, str]:
        """Get all settings as a dictionary"""
        settings = db.query(UserSettings).all()
        return {setting.setting_key: setting.setting_value for setting in settings}
    
    @staticmethod
    def get_path_settings(db: Session) -> Dict[str, Optional[str]]:
        """Get video and thumbnail directory settings"""
        return {
            "video_dir": SettingsService.get_setting(db, "custom_video_dir"),
            "thumbnail_dir": SettingsService.get_setting(db, "custom_thumbnail_dir")
        }
    
    @staticmethod
    def set_path_settings(db: Session, video_dir: Optional[str] = None, thumbnail_dir: Optional[str] = None) -> Dict[str, str]:
        """Set custom video and thumbnail directory paths"""
        result = {}
        
        if video_dir:
            SettingsService.set_setting(
                db, 
                "custom_video_dir", 
                video_dir, 
                "Custom video directory path"
            )
            result["video_dir"] = video_dir
        
        if thumbnail_dir:
            SettingsService.set_setting(
                db, 
                "custom_thumbnail_dir", 
                thumbnail_dir, 
                "Custom thumbnail directory path"
            )
            result["thumbnail_dir"] = thumbnail_dir
        
        return result
    
    @staticmethod
    def clear_path_settings(db: Session) -> bool:
        """Clear custom path settings to use defaults"""
        try:
            db.query(UserSettings).filter(
                UserSettings.setting_key.in_(["custom_video_dir", "custom_thumbnail_dir"])
            ).delete(synchronize_session=False)
            db.commit()
            return True
        except Exception:
            db.rollback()
            return False


settings_service = SettingsService()
