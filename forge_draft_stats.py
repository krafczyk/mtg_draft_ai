import jpype
from jpype import JImplements, JOverride
import jpype.imports
import os

# Set environment
os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-21-openjdk"
os.environ['DISPLAY'] = ":0.0"

# Start the JVM
jpype.startJVM(classpath=["/data0/matthew/Games/Forge/forge/forge-gui-desktop/target/forge-gui-desktop-2.0.01-SNAPSHOT-jar-with-dependencies.jar", "/data1/matthew/Projects/mtg_draft_ai/forge-helper/target/ForgeGuiDesktopHelper-1.0-SNAPSHOT.jar"])

from forge.gui import GuiBase
from forge import Singletons
ForgeGuiDesktopHelper = jpype.JClass("com.krafczyk.forge.ForgeGuiDesktopHelper")

GuiBase.setInterface(ForgeGuiDesktopHelper("/data0/matthew/Games/Forge/forge/forge-gui/"))

from forge.localinstance.properties import ForgeConstants

print(ForgeConstants.CACHE_CARD_PICS_DIR)
print(ForgeConstants.CACHE_ICON_PICS_DIR)
print(ForgeConstants.DEFAULT_SKINS_DIR)

Singletons.initializeOnce(True)
