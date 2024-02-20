#!/bin/bash


# Set the two arguments
GENERATOR="TinyLlama"
REVIEWER="Mixtral"

# Loop from 1 to 10
for i in {1..5}
do
    python RateStoriesNewPro.py $i $GENERATOR $REVIEWER
done



# Set the two arguments
GENERATOR="Mistral"
REVIEWER="Mixtral"

# Loop from 1 to 10
for i in {1..5}
do
    python RateStoriesNewPro.py $i $GENERATOR $REVIEWER
done



# Set the two arguments
GENERATOR="Nous-Capybara"
REVIEWER="Mixtral"

# Loop from 1 to 10
for i in {1..5}
do
    python RateStoriesNewPro.py $i $GENERATOR $REVIEWER
done