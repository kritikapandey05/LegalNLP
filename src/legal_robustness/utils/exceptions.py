class LegalRobustnessError(Exception):
    """Base exception for project-specific failures."""


class ConfigurationError(LegalRobustnessError):
    """Raised when configuration loading or validation fails."""


class DatasetPathError(LegalRobustnessError):
    """Raised when a dataset path is missing or invalid."""


class InspectionError(LegalRobustnessError):
    """Raised when dataset inspection fails for a supported file."""


class DatasetDiscoveryError(LegalRobustnessError):
    """Raised when dataset file discovery fails."""


class DatasetSchemaError(LegalRobustnessError):
    """Raised when a dataset file schema is missing required fields."""


class DatasetLoadError(LegalRobustnessError):
    """Raised when a dataset loader cannot safely continue."""


class DatasetNormalizationError(LegalRobustnessError):
    """Raised when normalization cannot safely continue."""


class DatasetReconstructionError(LegalRobustnessError):
    """Raised when section mapping or reconstruction cannot safely continue."""


class SectionTransferError(LegalRobustnessError):
    """Raised when RR-to-CJPE section-transfer preparation or modeling cannot safely continue."""


class PredictionError(LegalRobustnessError):
    """Raised when CJPE judgment-prediction baseline preparation or training fails."""


class PerturbationError(LegalRobustnessError):
    """Raised when section-aware perturbation generation or export fails."""
