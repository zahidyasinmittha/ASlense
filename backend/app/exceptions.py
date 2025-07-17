"""Custom exceptions for ASLense application"""


class ASLenseException(Exception):
    """Base exception for ASLense application"""
    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class VideoProcessingError(ASLenseException):
    """Exception raised when video processing fails"""
    pass


class CSVImportError(ASLenseException):
    """Exception raised when CSV import fails"""
    pass


class FileNotFoundError(ASLenseException):
    """Exception raised when required file is not found"""
    pass


class ConfigurationError(ASLenseException):
    """Exception raised when configuration is invalid"""
    pass


class ValidationError(ASLenseException):
    """Exception raised when data validation fails"""
    pass
