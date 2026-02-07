import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from io import StringIO
from gphyt.train.utils.logger import setup_logger


class TestSetupLogger:
    def test_setup_logger_basic(self):
        stream = StringIO()
        logger = setup_logger(
            name="test_logger",
            log_level=logging.INFO,
            stream=stream
        )

        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_setup_logger_with_file(self):
        with TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            logger = setup_logger(
                name="file_logger",
                log_file=log_file,
                stream=None
            )

            logger.info("Test message")

            assert logger.name == "file_logger"
            assert len(logger.handlers) == 1
            assert isinstance(logger.handlers[0], logging.FileHandler)
            assert log_file.exists()
            assert "Test message" in log_file.read_text()

    def test_setup_logger_custom_format(self):
        stream = StringIO()
        custom_format = "%(name)s - %(levelname)s - %(message)s"
        logger = setup_logger(
            name="custom_logger",
            stream=stream,
            format_string=custom_format
        )

        logger.info("Test message")
        output = stream.getvalue()

        assert "custom_logger - INFO - Test message" in output