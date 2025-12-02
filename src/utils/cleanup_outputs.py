import os
import shutil
from pathlib import Path
from loguru import logger

def cleanup():
    root = Path("./outputs")
    if not root.exists():
        logger.info("No outputs folder to clean.")
        return

    logger.info(f"Scanning {root} for empty or old folders...")

    removed_count = 0
    # Walk bottom-up so we can delete child directories then parents
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        current_dir = Path(dirpath)

        # Delete if directory is empty
        if not any(current_dir.iterdir()):
            try:
                current_dir.rmdir()
                logger.debug(f"Removed empty: {current_dir}")
                removed_count += 1
            except OSError:
                pass

    logger.success(f"Cleanup complete. Removed {removed_count} empty directories.")

if __name__ == "__main__":
    cleanup()