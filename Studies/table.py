#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:09:57 2020

@author: ghjuliasialelli

From .npy to Latex-like table
"""

#maxB =  {'100P_13': 243.37812150390624, '4P_SHAP_13': 367.2565527246094, '4P_RANDOM_13': 737.4790563867188, '100P_18': 38.10762524414062, '4P_SHAP_18': 62.68223614990234, '4P_RANDOM_18': 156.80763979003908}
#minB =  {'100P_13': 0.02521512695312822, '4P_SHAP_13': 0.00395802246093524, '4P_RANDOM_13': 0.4036223632812721, '100P_18': 12.77446903808594, '4P_SHAP_18': 0.26981796386719736, '4P_RANDOM_18': 64.67692309570312}
#stdB =  {'100P_13': 25.60607742831968, '4P_SHAP_13': 54.03290304588319, '4P_RANDOM_13': 132.51755170078528, '100P_18': 6.915331716975731, '4P_SHAP_18': 11.924291162891183, '4P_RANDOM_18': 26.315116535698298}
#listofdps = [13,18]

maxB =  {'4P_SHAP_drag_for_1': 25.97163705871582, '2P1V1N_drag_for_1': 35.060440117187504, '4P_SHAP_lift_for_1': 89.88308753051759, '2P1V1N_lift_for_1': 93.9881557373047, '4P_SHAP_angle_atta_1': 7.434747467041015, '2P1V1N_angle_atta_1': 19.869458345174788, '4P_SHAP_inlet_velocity_1': 2.4373054376220686, '2P1V1N_inlet_velocity_1': 20.179109150390623, '4P_SHAP_drag_for_2': 28.531586488647463, '2P1V1N_drag_for_2': 42.90114720718384, '4P_SHAP_lift_for_2': 95.28685954345701, '2P1V1N_lift_for_2': 88.76774409545897, '4P_SHAP_angle_atta_2': 7.2775078582763655, '2P1V1N_angle_atta_2': 15.129358558654786, '4P_SHAP_inlet_velocity_2': 2.250196244506835, '2P1V1N_inlet_velocity_2': 18.629884296874998}
minB =  {'4P_SHAP_drag_for_1': 0.0009290534973144249, '2P1V1N_drag_for_1': 0.00029861801147479383, '4P_SHAP_lift_for_1': 0.008667519531250889, '2P1V1N_lift_for_1': 0.0015800384521487132, '4P_SHAP_angle_atta_1': 3.940582275419047e-05, '2P1V1N_angle_atta_1': 0.000838546752929048, '4P_SHAP_inlet_velocity_1': 0.00039405151367333247, '2P1V1N_inlet_velocity_1': 0.0004645507812508498, '4P_SHAP_drag_for_2': 0.00019785667419436326, '2P1V1N_drag_for_2': 0.0020536904144285995, '4P_SHAP_lift_for_2': 0.0017866772460948255, '2P1V1N_lift_for_2': 0.0011287084960933669, '4P_SHAP_angle_atta_2': 0.003968296051025444, '2P1V1N_angle_atta_2': 0.0005891799926764918, '4P_SHAP_inlet_velocity_2': 6.531311035118392e-05, '2P1V1N_inlet_velocity_2': 0.00016540954589672197}
stdB =  {'4P_SHAP_drag_for_1': 1.7852514453698196, '2P1V1N_drag_for_1': 2.060543062517707, '4P_SHAP_lift_for_1': 6.60391572976609, '2P1V1N_lift_for_1': 6.298131428825914, '4P_SHAP_angle_atta_1': 0.9965908617208669, '2P1V1N_angle_atta_1': 1.1713881905510033, '4P_SHAP_inlet_velocity_1': 0.42143719834420806, '2P1V1N_inlet_velocity_1': 1.6460854353379946, '4P_SHAP_drag_for_2': 1.7927627654173646, '2P1V1N_drag_for_2': 2.3013392168445135, '4P_SHAP_lift_for_2': 6.6647823888685975, '2P1V1N_lift_for_2': 6.339173541854424, '4P_SHAP_angle_atta_2': 0.9932385617751096, '2P1V1N_angle_atta_2': 1.100995785211092, '4P_SHAP_inlet_velocity_2': 0.43006277589418457, '2P1V1N_inlet_velocity_2': 1.5898097190082918}

maxB =  {'4P_SHAP_1': 554.414756965332, '4P_SHAP_2': 552.9762107128906}
minB =  {'4P_SHAP_1': 0.06342129394531071, '4P_SHAP_2': 0.001463220214844796}
stdB =  {'4P_SHAP_1': 54.94544133742216, '4P_SHAP_2': 55.409630080273416}


listofdps = [1,2]

# ['drag_for', 'lift_for', 'angle_atta', 'inlet_velocity']
for output in ['pressure'] : 
    for dp in listofdps :
        
        print('------')
        print(output, dp)
        
        
        temp = """Model & Max Bias & Min Bias & Standard Deviation \\\ [0.5ex] 
             \hline\hline
             4P_SHAP & {:.3f} & {:.4f} &  {:.3f} \\\ """.format(maxB['4P_SHAP_{}'.format(dp)], minB['4P_SHAP_{}'.format(dp)], stdB['4P_SHAP_{}'.format(dp)])
        
        print(temp)
        
        print('------')


''' if wavenet 
format(maxB['100P_{}_{}'.format(output, dp)], minB['100P_{}_{}'.format(output, dp)], stdB['100P_{}_{}'.format(output, dp)], maxB['2P1V1N_{}_{}'.format(output, dp)], minB['2P1V1N_{}_{}'.format(output, dp)], stdB['2P1V1N_{}_{}'.format(output, dp)], maxB['4P_SHAP_{}_{}'.format(output, dp)], minB['4P_SHAP_{}_{}'.format(output, dp)], stdB['4P_SHAP_{}_{}'.format(output, dp)],maxB['4P_RANDOM_{}_{}'.format(output, dp)], minB['4P_RANDOM_{}_{}'.format(output, dp)], stdB['4P_RANDOM_{}_{}'.format(output, dp)])

if TCN :
    format(maxB['100P_{}'.format(dp)], minB['100P_{}'.format(dp)], stdB['100P_{}'.format(dp)], maxB['4P_SHAP_{}'.format(dp)], minB['4P_SHAP_{}'.format(dp)], stdB['4P_SHAP_{}'.format(dp)],maxB['4P_RANDOM_{}'.format(dp)], minB['4P_RANDOM_{}'.format(dp)], stdB['4P_RANDOM_{}'.format(dp)])

'''

    