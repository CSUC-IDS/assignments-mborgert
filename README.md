# Course Assignment Repo for Advanced Topics in Data Science for Spring 18

MITCH BORGERT

## R4DS HW 1 (Robin)
* Include the md file when you submit so that it can be viewed in github. 
* Compile to make sure it works before submission
* Write out the question text for the ones you are answering. Noone reading your work can tell what you're supposed to be doing. 
* A few items are missing. 
* Comment out - or don't include, code in your RMD file that opens windows when compiled (i.e. ?help)

## R4DS HW 2 (Robin)
* (Ref problem 32.4.5.1) Models of the formulation y ~ x -1 are done to remove the intercept term. Compare the output between `mod1` and `mod2`. `mod1` basically is an ANOVA. The estimates are the means. In the linear model formulation, the intercept is the baseline (reference group), and all other estimates are _in comparison to_ the reference group. So xb = 1.15 + 6.96 = 8.11. It's the same model, just framed differently, which is why you get the same predictions. 
* Ref 23.4.5 3. I'll look at this problem later. Did anyone else do this question? 

## ISLR Ch 2(Robin)
* Exploring boston data - always put lowess trend lines on scatterplots. 
* also use `gridExtra::grid.arrange` to put multiple plots on the same plot window. Or you could look at the `psych::pairs` function.
* answer questions in english, not code or variable names. i.e. _The median pupil-teacher ratio is 19.05_ instead of using `ptratio`

## ISLR Ch 3 (Robin)
* You need to comment on how the coefficient for `nox` is so far different than all the rest. 
* In general you need to comment on how the results from the individual SLR and the MLR regression coefficients differ. 
