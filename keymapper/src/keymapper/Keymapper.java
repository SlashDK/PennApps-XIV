/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package keymapper;

import javax.swing.JFrame;

/**
 *
 * @author fernando.bessa
 */
public class Keymapper {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
    JFrame.setDefaultLookAndFeelDecorated(true);
    mainwindow frame = new mainwindow();
    frame.setTitle("My First Swing Application");
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    frame.pack();
    frame.setVisible(true);
    }
    
}
