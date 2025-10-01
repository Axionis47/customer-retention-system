"""Data processors for all datasets."""
from data.processors.bank_processor import process_bank
from data.processors.oasst1_processor import process_oasst1
from data.processors.preferences_processor import process_preferences
from data.processors.telco_processor import process_telco

__all__ = ["process_telco", "process_bank", "process_oasst1", "process_preferences"]

