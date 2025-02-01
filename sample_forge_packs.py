import jpype
import jpype.imports
import os
import sys
import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample forge packs")
    parser.add_argument("--edition", type=str, help="The edition to sample packs from")
    parser.add_argument("--data-dir", type=str, help="The directory to get seventeen lands data from")
    parser.add_argument("--num-packs", type=int, help="The number of packs to sample")
    parser.add_argument("--output-file", type=str, help="The file to output the sampled packs to")
    args = parser.parse_args()

    # Set up JPype Environment
    os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-21-openjdk"
    os.environ['DISPLAY'] = ":0.0"

    # Start the JVM
    jpype.startJVM(
        "-Xms512m", "-Xmx4g",
        classpath=["/data0/matthew/Games/Forge/forge/forge-gui-desktop/target/forge-gui-desktop-2.0.02-SNAPSHOT-jar-with-dependencies.jar", "/data1/matthew/Projects/mtg_draft_ai/forge-helper/target/ForgeGuiDesktopHelper-1.0-SNAPSHOT.jar"],
        convertStrings=True,
    )

    from forge.gui import GuiBase
    from forge import Singletons
    ForgeGuiDesktopHelper = jpype.JClass("com.krafczyk.forge.ForgeGuiDesktopHelper")

    # Start up Gui
    GuiBase.setInterface(ForgeGuiDesktopHelper("/data0/matthew/Games/Forge/forge/forge-gui/"))

    # Initialize Singletons
    Singletons.initializeOnce(False)

    from forge.model import FModel
    edition = "FDN"

    # Prepare useful variables
    from seventeenlands_datasets import get_data_dir_dtypes
    import pandas as pd

    dtypes = get_data_dir_dtypes(args.data_dir)
    columns = dtypes.keys()

    # Get only the pack_card_* columns
    card_cols = [ col for col in columns if col.startswith("pack_card_") ]
    # Strip the "pack_card_" prefix
    card_cols = [ col[len("pack_card_"):] for col in card_cols ]

    # Method to create a series of zeros indexed by card name.
    def create_pack_series(card_cols: list[str]) -> pd.Series:
        return pd.Series(0, index=card_cols)

    # Sample cards using forge's generator
    from forge.item.generation import BoosterGenerator

    pack_rows = []
    # Iterate using tqdm progress bar
    for _ in tqdm.tqdm(range(args.num_packs)):
        booster_pack = BoosterGenerator.getBoosterPack(FModel.getMagicDb().getBoosters().get(edition))

        pack = create_pack_series(card_cols)
        for card in booster_pack:
            pack[card.getName()] += 1

        # Append pack to row list
        pack_rows.append(pack)

    # Create and write the card data
    pd.DataFrame(pack_rows, columns=card_cols).to_csv(args.output_file, index=False)

    # Shut down the JVM
    jpype.shutdownGuiEnvironment()
    jpype.shutdownJVM()

    sys.exit(0)
