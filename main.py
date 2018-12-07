#help("modules")
import petronautics.models.petroleumengineering as pe
obj = pe.BuckleyLeverett('buckley_leverett_input.csv')
obj.Run()
obj.Plot()
