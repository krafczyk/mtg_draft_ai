import os
import jpype
import jpype.imports
import glob


def init_bridge(
        java_home:str = "/usr/lib/jvm/java-21-openjdk",
        forge_classpath:str = "/data0/matthew/Games/Forge/forge/forge-gui-desktop/target/forge-gui-desktop-*-jar-with-dependencies.jar",
        forge_helper_classpath:str = "/data1/matthew/Projects/mtg_draft_ai/forge-helper/target/ForgeGuiDesktopHelper-*-SNAPSHOT.jar",
        forge_gui_dir:str = "/data0/matthew/Games/Forge/forge/forge-gui/",
        display:str = ":0.0",
        jpype_args: list[str]|None = None,
        jpype_kwargs: dict[str,str|bool]|None=None,
        verbose: bool = False):

    # Set up JPype Environment
    os.environ['JAVA_HOME'] = java_home
    os.environ['DISPLAY'] = display

    if jpype_args is None:
        jpype_args = ["-Xms512m", "-Xmx4g"]

    if jpype_kwargs is None:
        jpype_kwargs = {'convertStrings': True}

    glob_results = glob.glob(forge_classpath)
    if len(glob_results) == 0:
        raise FileNotFoundError(f"No files found matching forge_classpath pattern: {forge_classpath}")
    resolved_forge_classpath = glob_results[0]
    glob_results = glob.glob(forge_helper_classpath)
    if len(glob_results) == 0:
        raise FileNotFoundError(f"No files found matching forge_helper_classpath pattern: {forge_helper_classpath}")
    resolved_forge_helper_classpath = glob_results[0]

    if verbose:
        print(f"Resolved forge_classpath to: {resolved_forge_classpath}")
        print(f"Resolved forge_helper_classpath to: {resolved_forge_helper_classpath}")

    # Start the JVM
    jpype.startJVM(
        *jpype_args,
        classpath=[resolved_forge_classpath, resolved_forge_helper_classpath],
        **jpype_kwargs,
    )

    from forge.gui import GuiBase
    from forge import Singletons
    ForgeGuiDesktopHelper = jpype.JClass("com.krafczyk.forge.ForgeGuiDesktopHelper")

    # Start up Gui
    GuiBase.setInterface(ForgeGuiDesktopHelper(forge_gui_dir))

    # Initialize Singletons
    Singletons.initializeOnce(False)
