package com.krafczyk.forge;

import forge.GuiDesktop;

public class ForgeGuiDesktopHelper extends GuiDesktop {
    private String assetsPath;

    // Constructor that takes additional argument for the path
    public ForgeGuiDesktopHelper(String assetsPath) {
        super();  // Call the superclass constructor
        this.assetsPath = assetsPath;
    }

    @Override
    public String getAssetsDir() {
        // Use the stored path instead of the hardcoded value
        return assetsPath;
    }
}
