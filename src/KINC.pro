
# Default setting for GUI
isEmpty(GUI) { GUI = "yes" }

# Basic settings
TEMPLATE = subdirs

# Subdir projects
SUBDIRS += \
    core \
    cli

# Dependencies
cli.depends = core

# This is if GUI is enabled
equals(GUI,"yes") {
    # Add GUI project and dependencies
    SUBDIRS += gui
    gui.depends = core
}
