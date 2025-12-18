/*
- CAUTION SAVING AND LOADING IS OPTIMIZED TO WORK BASED ON WHAT ACTIVATION-FUNCTIONS, FLOAT\DOUBLE\INT-MODE OR BIAS-MODE YOU HAVE DEFINED (OR NOT DEFINED AT ALL)
- CAUTION SAVING AND LOADING IS OPTIMIZED TO WORK BASED ON WHAT ACTIVATION-FUNCTIONS, FLOAT\DOUBLE\INT-MODE OR BIAS-MODE YOU HAVE DEFINED (OR NOT DEFINED AT ALL)
- CAUTION SAVING AND LOADING IS OPTIMIZED TO WORK BASED ON WHAT ACTIVATION-FUNCTIONS, FLOAT\DOUBLE\INT-MODE OR BIAS-MODE YOU HAVE DEFINED (OR NOT DEFINED AT ALL)
- ^^^^^^^ THE ABOVE STATEMENT IS ALSO TRUE WHETHER YOU USED REDUCE_RAM_WEIGHTS_LVL2 OR NOT DURING TRAINING !!!!
- CAUTION THIS EXAMPLE BARELY FITS THE RAM OF ARDUINO UNO. EXPIRIMENT WITH CAUTION ON IT* | THIS IS MAINLY THE RESULT OF NOT USING B01000000 RAM-OPTIMIZATION ...
*/
#define NumberOf(arg) ((unsigned int) (sizeof (arg) / sizeof (arg [0]))) // calculates the number of layers (in this case 3)
#include <stdio.h>             // for printf
#include <fstream>
#include "sleep.h"
#define FILENAME "./WEIGHTS2.BIN"
#define _3_OPTIMIZE 0B00000010 // Enable native fs support
#define _1_OPTIMIZE 0B00011000 // https://github.com/GiorgosXou/NeuralNetworks#define-macro-properties
#define ACTIVATION__PER_LAYER  // DEFAULT KEYWORD for allowing the use of any Activation-Function per "Layer-to-Layer".
        #define Sigmoid // 0     Says to the compiler to compile the Sigmoid Activation-Function 
        #define Tanh    // 1     Says to the compiler to compile the Tanh    Activation-Function 

#include "../NeuralNetwork.h"

NeuralNetwork *NN;

unsigned int layers[] = {3, 7, 1}; // 3 layers: (1st)layer with 3-inputs/features (2nd)layer 7-hidden-neurons and (3rd)layer with 1-output-neuron
byte Actv_Functions[] = {   1, 0}; // 1 = Tanh and 0 = Sigmoid (just as a proof of consept)

float *output; // 3th layer's output
std::fstream myFile;

//Default Inputs/Training-Data
const float inputs[8][3] = {
  {0, 0, 0}, // = 0
  {0, 0, 1}, // = 1
  {0, 1, 0}, // = 1
  {0, 1, 1}, // = 0
  {1, 0, 0}, // = 1
  {1, 0, 1}, // = 0
  {1, 1, 0}, // = 0
  {1, 1, 1}  // = 1
};
const float expectedOutput[8][1] = {{0}, {1}, {1}, {0}, {1}, {0}, {0}, {1}}; // values that we are expecting to get from the 4th/(output)layer of the Neuralnetwork, in other words something like a feedback to the Neural-network.



void train_and_save_NN()
{
  NN = new NeuralNetwork(layers,NumberOf(layers),Actv_Functions); //Initialization of NeuralNetwork object
  printf("Training the NN\n");
  do{
    for (unsigned int j = 0; j < NumberOf(inputs); j++) // Epoch
    {
      NN->FeedForward(inputs[j]);      // FeedForwards the input arrays through the NN | stores the output array internally
      NN->BackProp(expectedOutput[j]); // "Tells" to the NN if the output was the-expected-correct one | then, "teaches" it
    }

    // Prints the MSError.
    printf("MSE: %f\n", NN->MeanSqrdError);

    // Loops through each epoch Until MSE goes  < 0.003
  }while(NN->getMeanSqrdError(NumberOf(inputs)) > 0.003);
  
  myFile.open(FILENAME, std::ios::binary | std::fstream::out);
  NN->save(myFile); // Saves the NN into the FILENAME
  myFile.close();
  printf("Done\n");
}


float input[3]; // Input Array
void loop() // testing a hypotherical scenarion
{
  //For example: here you could have a live input from two buttons/switches (or a feed from a sensor if it was a different NN)
  input[0] = NN_RANDOM(0,2); // ... lets say input from a       button/Switch | random(2) = 0 or 1
  input[1] = NN_RANDOM(0,2); // ... lets say input from another button/Switch | random(2) = 0 or 1
  input[2] = NN_RANDOM(0,2); // ... lets say input from another button/Switch | random(2) = 0 or 1
  output = NN->FeedForward(input); // FeedForwards the input-array through the NN | returns the predicted output(s)

  printf("Inputs: %f, %f, %f\n", round( input[0]), round(input[1]), round(input[2])); // Although you can kind of use casting like...
  printf("Output: %f\n", round(output[0])); // ... "(int)output[0]" instead of round to reduces ROM usage, be careful
  //
  SLEEP(1500); loop();
}


int main()
{
  std::ifstream fin(FILENAME);
  if (fin){ // Checks if FILENAME exists
    myFile.open(FILENAME, std::ios::binary | std::ios::in);
    NN = new NeuralNetwork(myFile); // Loads NN | [ use NN->load("/my_file"); while running, to load a new one ]
    myFile.close();
  }else{
    train_and_save_NN();              // Else If it doesn't exist, trains it and saves it ...   
  }
  NN->print();                        // Prints the weights and biases of each layer
  while (true)
    loop();
}


