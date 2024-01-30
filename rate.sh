#!/bin/bash


# Set the two arguments
GENERATOR="TinyLlama"
REVIEWER="Mixtral"

# Loop from 1 to 10
for i in {1..10}
do
    python RateStories3.py $i $GENERATOR $REVIEWER
done

# Set the two arguments
GENERATOR="TinyLlama"
REVIEWER="Nous-Capybara"

# Loop from 1 to 10
for i in {1..10}
do
    python RateStories3.py $i $GENERATOR $REVIEWER
done




# Set the two arguments
GENERATOR="Mistral"
REVIEWER="Mixtral"

# Loop from 1 to 10
for i in {1..10}
do
    python RateStories3.py $i $GENERATOR $REVIEWER
done

# Set the two arguments
GENERATOR="Mistral"
REVIEWER="Nous-Capybara"

# Loop from 1 to 10
for i in {1..10}
do
    python RateStories3.py $i $GENERATOR $REVIEWER
done






# Set the two arguments
GENERATOR="Nous-Capybara"
REVIEWER="Mixtral"

# Loop from 1 to 10
for i in {1..10}
do
    python RateStories3.py $i $GENERATOR $REVIEWER
done



# Set the two arguments
GENERATOR="Nous-Capybara"
REVIEWER="Nous-Capybara"

# Loop from 1 to 10
for i in {1..10}
do
    python RateStories3.py $i $GENERATOR $REVIEWER
done



