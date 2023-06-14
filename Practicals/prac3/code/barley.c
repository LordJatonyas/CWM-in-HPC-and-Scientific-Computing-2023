/**************************************************
 *                                                *
 * First attempt at a code to calcule lost barley *
 * by A. Farmer                                   *
 * 18/05/18                                       *
 *                                                *
 **************************************************/

// Include any headers from the C standard library here
#include <stdio.h>
#include <stdlib.h>

// Define any constants that I need to use here
#define PI 3.141592

// This is where I should put my function prototypes
float area_of_circle(float radius); 

// Now I start my code with main()
int main() {

    // In here I need to declare my variables
    float *radii;
    float radius;
    int num;
    float total_area = 0;
    float loss_in_kg = 0;
    float length;
    float width;
    float farm_area;
    float percentage_loss;
    float monetary_loss_in_gbp;

    // Next I need to get input from the user.
    // I'll do this by using a printf() to ask the user to input the radii.
    scanf("%d", &num);
    scanf("%f", &length);
    scanf("%f", &width);
    farm_area = length * width;

    radii = (float *) malloc(2 * num * sizeof(float));
    for (int i = 0; i < num; i++) {
        // Even positions will be the inner radius, outer radius are odds
        scanf("%f", &radius);
        *(radii + i) = radius;
    }
    // Now I need to loop through the radii caluclating the area for each
    for (int j = 0; j < num; j = j + 2) {
        total_area = total_area + area_of_circle(*(radii + j + 1)) - area_of_circle(*(radii + j));
    }
    // Next I'll sum up all of the individual areas

    /******************************************************************
     *                                                                *
     * Now I know the total area I can use the following information: *
     *                                                                *
     * One square meter of crop produces about 135 grams of barley    *
     *                                                                *
     * One kg of barley sells for about 10 pence                      *
     *                                                                *
     ******************************************************************/

    // Using the above I'll work out how much barley has been lost.
    percentage_loss = total_area / farm_area * 100;
    loss_in_kg = total_area * 0.135;
    monetary_loss_in_gbp = loss_in_kg * 0.1;

    // Finally I'll use a printf() to print this to the screen.
    printf("\nTotal area lost in m^2 is:\t%f\n", total_area);
    printf("Total loss in kg is:\t\t%f\n", loss_in_kg);
    printf("Percentage loss is: \t\t%f\n", percentage_loss);
    printf("Monetary loss in GBP: \t\t%f\n", monetary_loss_in_gbp);

    free(radii);
    return(0);
}

// I'll put my functions here:
float area_of_circle(float radius) {
    float area = PI * radius * radius;
    return area;
}
