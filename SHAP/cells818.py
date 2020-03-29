#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 21:36:02 2020

@author: ghjuliasialelli

818 cells
"""

bad = 7581.49997

cells = [7067.5,
 7065.50002,
 7069.50003,
 7063.5,
 7071.5,
 7061.50002,
 7059.5,
 7073.49998,
 7057.4999099999995,
 7075.5,
 7055.5,
 7077.5001,
 7053.4998,
 7079.5,
 7051.5,
 7081.50025,
 7049.49968,
 7083.5,
 7047.5,
 7085.5004,
 7045.4996,
 7087.5,
 7043.5,
 7089.5005,
 7041.4996,
 7091.5,
 7039.5,
 7093.500620000001,
 7037.499470000001,
 7095.5,
 7035.5,
 7097.50074,
 7033.4993,
 7031.5,
 7099.5,
 7029.49908,
 7101.501009999999,
 7027.5,
 7103.5,
 7025.49885,
 7105.501359999999,
 7023.5,
 7107.5,
 7021.49862,
 7109.501740000001,
 7019.5,
 7111.5,
 7017.4984,
 7015.5,
 7113.501590000001,
 7013.49951,
 7115.5,
 7011.5,
 7117.49822,
 7009.50173,
 7119.5,
 7007.5,
 7005.5016,
 7121.4979299999995,
 7003.5,
 7123.5,
 7001.50139,
 7125.498259999999,
 6999.5,
 7127.5,
 6997.50115,
 6995.5,
 7129.498640000001,
 6993.50087,
 7131.5,
 6991.5,
 7133.49907,
 6989.5005599999995,
 7135.5,
 6987.5,
 6985.50025,
 7137.499559999999,
 6983.5,
 7139.5,
 6981.4999099999995,
 7141.50039,
 6979.5,
 6977.49955,
 7143.5,
 6975.5,
 7145.5016,
 6973.49915,
 7147.5,
 6971.5,
 6969.49871,
 7149.50292,
 6967.5,
 7151.5,
 6965.497759999999,
 6963.5,
 7153.50325,
 6961.497170000001,
 7155.5,
 6959.5,
 7157.500529999999,
 6957.49953,
 6955.5,
 7159.5,
 6953.5,
 7161.500140000001,
 6951.5,
 6949.5,
 7163.5,
 6947.5,
 7165.49978,
 6945.5,
 6943.5,
 7167.5,
 6941.50039,
 7169.497579999999,
 6939.5,
 6937.50262,
 7171.5,
 6935.5,
 7173.496999999999,
 6933.502579999999,
 6931.5,
 7175.5,
 6929.50222,
 7177.49664,
 6927.5,
 6925.50132,
 7179.5,
 6923.5,
 6921.50039,
 7181.4992,
 6919.5,
 7183.5,
 6917.49945,
 6915.5,
 7185.502170000001,
 6913.4985400000005,
 6911.5,
 7187.5,
 6909.49747,
 7189.5050599999995,
 6907.5,
 6905.4965,
 7191.5,
 6903.5,
 6901.501509999999,
 7193.49817,
 6899.5,
 7195.5,
 6897.50305,
 6895.5,
 7197.496359999999,
 6893.50181,
 6891.5,
 7199.5,
 6889.500529999999,
 6887.5,
 7201.50033,
 6885.499220000001,
 7203.5,
 6883.5,
 6881.49788,
 7205.50395,
 6879.5,
 6877.496590000001,
 7207.5,
 6875.5,
 6873.498329999999,
 7209.5021400000005,
 6871.5,
 6869.499629999999,
 7211.5,
 6867.5,
 6865.49995,
 7213.499720000001,
 6863.5,
 6861.50108,
 7215.5,
 6859.5,
 6857.50245,
 7217.496279999999,
 6855.5,
 6853.503229999999,
 7219.5,
 6851.5,
 6849.5023599999995,
 7221.497170000001,
 6847.5,
 6845.5005200000005,
 7223.5,
 6843.5,
 6841.49865,
 7225.50265,
 6839.5,
 6837.49675,
 7227.5,
 6835.5,
 6833.496190000001,
 7229.50231,
 6831.5,
 6829.503540000001,
 7231.5,
 6827.5,
 6825.50293,
 7233.49567,
 6823.5,
 6821.50105,
 7235.5,
 6819.5,
 6817.4991,
 6815.5,
 7237.50425,
 6813.496840000001,
 6811.5,
 7239.5,
 6809.49645,
 6807.5,
 7241.50262,
 6805.49917,
 6803.5,
 7243.5,
 6801.499690000001,
 6799.5,
 6797.500590000001,
 7245.49822,
 6795.5,
 6793.50442,
 7247.5,
 6791.5,
 6789.50328,
 6787.5,
 7249.49508,
 6785.501270000001,
 6783.5,
 7251.5,
 6781.49886,
 6779.5,
 7253.504690000001,
 6777.49642,
 6775.5,
 6773.496940000001,
 7255.5,
 6771.5,
 6769.50417,
 6767.5,
 7257.49695,
 6765.50261,
 6763.5,
 7259.5,
 6761.4999,
 6759.5,
 6757.497509999999,
 7261.5018900000005,
 6755.5,
 6753.496440000001,
 6751.5,
 7263.5,
 6749.497590000001,
 6747.5,
 7265.5018,
 6745.50026,
 6743.5,
 6741.502740000001,
 7267.5,
 6739.5,
 6737.50305,
 6735.5,
 7269.49597,
 6733.5029,
 6731.5,
 6729.50017,
 7271.5,
 6727.5,
 6725.49722,
 6723.5,
 7273.502329999999,
 6721.49545,
 6719.5,
 6717.503490000001,
 7275.5,
 6715.5,
 6713.50313,
 6711.5,
 7277.49708,
 6709.49992,
 6707.5,
 6705.49697,
 7279.5,
 6703.5,
 6701.49716,
 6699.5,
 6697.49762,
 7281.5021799999995,
 6695.5,
 6693.50006,
 6691.5,
 7283.5,
 6689.503970000001,
 6687.5,
 6685.50389,
 6683.5,
 7285.49925,
 6681.50133,
 6679.5,
 6677.497960000001,
 7287.5,
 6675.5,
 6673.4951,
 6671.5,
 6669.5024,
 7289.49943,
 6667.5,
 6665.503479999999,
 6663.5,
 7291.5,
 6661.499879999999,
 6659.5,
 6657.49729,
 6655.5,
 7293.4993,
 6653.49596,
 6651.5,
 6649.49862,
 6647.5,
 7295.5,
 6645.501429999999,
 6643.5,
 6641.50484,
 6639.5,
 7297.504359999999,
 6637.50273,
 6635.5,
 6633.49952,
 6631.5,
 6629.49551,
 7299.5,
 6627.5,
 6625.49915,
 6623.5,
 6621.5045,
 7301.49452,
 6619.5,
 6617.50047,
 6615.5,
 6613.49682,
 7303.5,
 6611.5,
 6609.49735,
 6607.5,
 6605.49842,
 6603.5,
 7305.50441,
 6601.501429999999,
 6599.5,
 6597.5041200000005,
 6595.5,
 6593.502829999999,
 7307.50001,
 6591.5,
 6589.49825,
 6587.5,
 6585.49491,
 6583.5,
 7309.497420000001,
 6581.50339,
 6579.5,
 6577.50247,
 6575.5,
 6573.49819,
 7311.49998,
 6571.5,
 6569.49687,
 6567.5,
 6565.49688,
 6563.5,
 7313.49835,
 6561.5001,
 6559.5,
 6557.50489,
 6555.5,
 6553.50361,
 6551.5,
 7315.49998,
 6549.49862,
 6547.5,
 6545.49492,
 6543.5,
 6541.50383,
 7317.501509999999,
 6539.5,
 6537.50187,
 6535.5,
 6533.49736,
 6531.5,
 6529.49715,
 7319.50003,
 6527.5,
 6525.49868,
 6523.5,
 6521.50224,
 6519.49998,
 6517.50416,
 7321.5012799999995,
 6515.50006,
 6513.49971,
 6511.50082,
 6509.48116,
 6507.50185,
 7323.5,
 10085.600559999999,
 10915.98877,
 8137.508640000001,
 8135.4992,
 8133.500209999999,
 8131.499940000001,
 7325.498740000001,
 8129.495859999999,
 8127.50002,
 8125.49777,
 8123.5,
 8121.5014,
 8119.5,
 7327.49997,
 8117.502829999999,
 8115.5,
 8113.50262,
 8111.5,
 8109.498059999999,
 7329.498479999999,
 8107.5,
 8105.496290000001,
 8103.5,
 8101.50508,
 8099.5,
 8097.5013,
 7331.50002,
 8095.5,
 8093.49632,
 8091.5,
 8089.4951599999995,
 8087.5,
 7333.50155,
 8085.499959999999,
 8083.5,
 8081.50315,
 8079.5,
 8077.50312,
 8075.5,
 7335.50002,
 8073.501759999999,
 8071.5,
 8069.49747,
 8067.5,
 8065.49672,
 7337.50269,
 8063.5,
 8061.505090000001,
 8059.5,
 8057.501690000001,
 8055.5,
 7339.49999,
 8053.497109999999,
 8051.5,
 8049.495940000001,
 8047.5,
 8045.4985799999995,
 7341.49555,
 8043.5,
 8041.50162,
 8039.5,
 8037.50267,
 8035.5,
 7343.5,
 8033.503140000001,
 8031.5,
 8029.499470000001,
 8027.5,
 7345.50542,
 8025.49545,
 8023.5,
 8021.501029999999,
 8019.5,
 7347.5,
 8017.504440000001,
 8015.5,
 8013.50042,
 8011.5,
 8009.497240000001,
 7349.4958,
 8007.5,
 8005.49515,
 8003.5,
 8001.498640000001,
 7351.5,
 7999.5,
 7997.501429999999,
 7995.5,
 7993.50403,
 7353.500529999999,
 7991.5,
 7989.5027,
 7987.5,
 7355.5,
 7985.50007,
 7983.5,
 7981.49647,
 7979.5,
 7357.5008,
 7977.49771,
 7975.5,
 7973.50491,
 7971.5,
 7359.5,
 7969.501990000001,
 7967.5,
 7965.49862,
 7361.500540000001,
 7963.5,
 7961.49608,
 7959.5,
 7957.49608,
 7363.5,
 7955.5,
 7953.49998,
 7951.5,
 7365.4979299999995,
 7949.502409999999,
 7947.5,
 7945.50285,
 7367.5,
 7943.5,
 7941.503000000001,
 7939.5,
 7937.500040000001,
 7369.502829999999,
 7935.5,
 7933.49682,
 7931.5,
 7371.5,
 7929.49662,
 7927.5,
 7925.50457,
 7373.49772,
 7923.5,
 7921.502740000001,
 7919.5,
 7375.5,
 7917.499790000001,
 7915.5,
 7913.49707,
 7377.50402,
 7911.5,
 7909.4969599999995,
 7907.5,
 7379.5,
 7905.49729,
 7903.5,
 7901.49976,
 7381.49822,
 7899.5,
 7897.50247,
 7383.5,
 7895.5,
 7893.503540000001,
 7891.5,
 7385.4981099999995,
 7889.50247,
 7887.5,
 7885.500059999999,
 7387.5,
 7883.5,
 7881.497359999999,
 7389.50313,
 7879.5,
 7877.49585,
 7875.5,
 7391.5,
 7873.503159999999,
 7871.5,
 7393.495209999999,
 7869.50355,
 7867.5,
 7865.501109999999,
 7395.5,
 7863.5,
 7861.4987,
 7397.5048799999995,
 7859.5,
 7857.496690000001,
 7855.5,
 7399.5,
 7853.495620000001,
 7851.5,
 7401.5018,
 7849.49943,
 7847.5,
 7845.50032,
 7403.5,
 7843.5,
 7841.50085,
 7405.49746,
 7839.5,
 7837.50358,
 7407.5,
 7835.5,
 7833.50313,
 7409.49568,
 7831.5,
 7829.50087,
 7827.5,
 7411.5,
 7825.49892,
 7823.5,
 7413.5043,
 7821.49704,
 7819.5,
 7415.5,
 7817.4965,
 7815.5,
 7417.49781,
 7813.503879999999,
 7811.5,
 7419.5,
 7809.5032200000005,
 7807.5,
 7421.497240000001,
 7805.50133,
 7803.5,
 7423.5,
 7801.499459999999,
 7799.5,
 7425.5028,
 7797.49762,
 7795.5,
 7427.5,
 7793.496770000001,
 7791.5,
 7429.503740000001,
 7789.49756,
 7787.5,
 7431.5,
 7785.49895,
 7783.5,
 7433.50035,
 7781.50005,
 7779.5,
 7435.5,
 7777.500370000001,
 7775.5,
 7437.4979,
 7773.50171,
 7771.5,
 7439.5,
 7769.5034,
 7767.5,
 7441.49601,
 7765.5021,
 7763.5,
 7443.5,
 7761.500759999999,
 7445.49961,
 7759.5,
 7757.49945,
 7447.5,
 7755.5,
 7753.49817,
 7449.50359,
 7751.5,
 7749.496929999999,
 7451.5,
 7747.5,
 7453.501940000001,
 7745.498559999999,
 7743.5,
 7455.5,
 7741.50352,
 7739.5,
 7457.494959999999,
 7737.50252,
 7459.5,
 7735.5,
 7733.501440000001,
 7461.49779,
 7731.5,
 7729.500540000001,
 7463.5,
 7727.5,
 7465.500759999999,
 7725.4996,
 7723.5,
 7467.5,
 7721.49867,
 7469.50333,
 7719.5,
 7717.49777,
 7471.5,
 7715.5,
 7713.497420000001,
 7473.50302,
 7711.5,
 7475.5,
 7709.49739,
 7707.5,
 7477.50245,
 7705.499640000001,
 7479.5,
 7703.5,
 7701.5,
 7481.50023,
 7699.5,
 7483.5,
 7697.5,
 7695.5,
 7485.499859999999,
 7693.5,
 7487.5,
 7691.5,
 7689.50051,
 7489.4994799999995,
 7687.5,
 7491.5,
 7685.502840000001,
 7493.4967799999995,
 7683.5,
 7681.50223,
 7495.5,
 7679.5,
 7497.49706,
 7677.50129,
 7499.5,
 7675.5,
 7673.50085,
 7501.498390000001,
 7671.5,
 7503.5,
 7669.50045,
 7667.5,
 7505.49959,
 7665.5000900000005,
 7507.5,
 7663.5,
 7509.50043,
 7661.49975,
 7659.5,
 7511.5,
 7657.49943,
 7513.50092,
 7655.5,
 7515.5,
 7653.49913,
 7517.501359999999,
 7651.5,
 7649.49885,
 7519.5,
 7647.5,
 7521.501740000001,
 7645.4986,
 7523.5,
 7643.5,
 7525.5020700000005,
 7641.4984,
 7639.5,
 7527.5,
 7637.49828,
 7529.5018,
 7635.5,
 7531.5,
 7633.5005200000005,
 7533.49845,
 7631.5,
 7629.5016,
 7535.5,
 7627.5,
 7537.49825,
 7625.50138,
 7539.5,
 7623.5,
 7541.49863,
 7621.50114,
 7543.5,
 7619.5,
 7545.498979999999,
 7617.50091,
 7547.5,
 7615.5,
 7613.5007,
 7549.49925,
 7611.5,
 7551.5,
 7609.500529999999,
 7553.499379999999,
 7607.5,
 7555.5,
 7605.5004,
 7557.4995,
 7603.5,
 7559.5,
 7601.5004,
 7561.4996,
 7599.5,
 7563.5,
 7597.50032,
 7565.49975,
 7595.5,
 7567.5,
 7593.5002,
 7569.499890000001,
 7591.5,
 7571.5,
 7589.50008,
 7573.50002,
 7587.5,
 7575.5,
 7585.49998,
 7577.49998,
 7583.5,
 7579.5]





old_cells = [ 6507.50185,  6509.48116,  6511.50082,  6513.49971,  6515.50006,
        6517.50416,  6519.49998,  6521.50224,  6523.5    ,  6525.49868,
        6527.5    ,  6529.49715,  6531.5    ,  6533.49736,  6535.5    ,
        6537.50187,  6539.5    ,  6541.50383,  6543.5    ,  6545.49492,
        6547.5    ,  6549.49862,  6551.5    ,  6553.50361,  6555.5    ,
        6557.50489,  6559.5    ,  6561.5001 ,  6563.5    ,  6565.49688,
        6567.5    ,  6569.49687,  6571.5    ,  6573.49819,  6575.5    ,
        6577.50247,  6579.5    ,  6581.50339,  6583.5    ,  6585.49491,
        6587.5    ,  6589.49825,  6591.5    ,  6593.50283,  6595.5    ,
        6597.50412,  6599.5    ,  6601.50143,  6603.5    ,  6605.49842,
        6607.5    ,  6609.49735,  6611.5    ,  6613.49682,  6615.5    ,
        6617.50047,  6619.5    ,  6621.5045 ,  6623.5    ,  6625.49915,
        6627.5    ,  6629.49551,  6631.5    ,  6633.49952,  6635.5    ,
        6637.50273,  6639.5    ,  6641.50484,  6643.5    ,  6645.50143,
        6647.5    ,  6649.49862,  6651.5    ,  6653.49596,  6655.5    ,
        6657.49729,  6659.5    ,  6661.49988,  6663.5    ,  6665.50348,
        6667.5    ,  6669.5024 ,  6671.5    ,  6673.4951 ,  6675.5    ,
        6677.49796,  6679.5    ,  6681.50133,  6683.5    ,  6685.50389,
        6687.5    ,  6689.50397,  6691.5    ,  6693.50006,  6695.5    ,
        6697.49762,  6699.5    ,  6701.49716,  6703.5    ,  6705.49697,
        6707.5    ,  6709.49992,  6711.5    ,  6713.50313,  6715.5    ,
        6717.50349,  6719.5    ,  6721.49545,  6723.5    ,  6725.49722,
        6727.5    ,  6729.50017,  6731.5    ,  6733.5029 ,  6735.5    ,
        6737.50305,  6739.5    ,  6741.50274,  6743.5    ,  6745.50026,
        6747.5    ,  6749.49759,  6751.5    ,  6753.49644,  6755.5    ,
        6757.49751,  6759.5    ,  6761.4999 ,  6763.5    ,  6765.50261,
        6767.5    ,  6769.50417,  6771.5    ,  6773.49694,  6775.5    ,
        6777.49642,  6779.5    ,  6781.49886,  6783.5    ,  6785.50127,
        6787.5    ,  6789.50328,  6791.5    ,  6793.50442,  6795.5    ,
        6797.50059,  6799.5    ,  6801.49969,  6803.5    ,  6805.49917,
        6807.5    ,  6809.49645,  6811.5    ,  6813.49684,  6815.5    ,
        6817.4991 ,  6819.5    ,  6821.50105,  6823.5    ,  6825.50293,
        6827.5    ,  6829.50354,  6831.5    ,  6833.49619,  6835.5    ,
        6837.49675,  6839.5    ,  6841.49865,  6843.5    ,  6845.50052,
        6847.5    ,  6849.50236,  6851.5    ,  6853.50323,  6855.5    ,
        6857.50245,  6859.5    ,  6861.50108,  6863.5    ,  6865.49995,
        6867.5    ,  6869.49963,  6871.5    ,  6873.49833,  6875.5    ,
        6877.49659,  6879.5    ,  6881.49788,  6883.5    ,  6885.49922,
        6887.5    ,  6889.50053,  6891.5    ,  6893.50181,  6895.5    ,
        6897.50305,  6899.5    ,  6901.50151,  6903.5    ,  6905.4965 ,
        6907.5    ,  6909.49747,  6911.5    ,  6913.49854,  6915.5    ,
        6917.49945,  6919.5    ,  6921.50039,  6923.5    ,  6925.50132,
        6927.5    ,  6929.50222,  6931.5    ,  6933.50258,  6935.5    ,
        6937.50262,  6939.5    ,  6941.50039,  6943.5    ,  6945.5    ,
        6947.5    ,  6949.5    ,  6951.5    ,  6953.5    ,  6955.5    ,
        6957.49953,  6959.5    ,  6961.49717,  6963.5    ,  6965.49776,
        6967.5    ,  6969.49871,  6971.5    ,  6973.49915,  6975.5    ,
        6977.49955,  6979.5    ,  6981.49991,  6983.5    ,  6985.50025,
        6987.5    ,  6989.50056,  6991.5    ,  6993.50087,  6995.5    ,
        6997.50115,  6999.5    ,  7001.50139,  7003.5    ,  7005.5016 ,
        7007.5    ,  7009.50173,  7011.5    ,  7013.49951,  7015.5    ,
        7017.4984 ,  7019.5    ,  7021.49862,  7023.5    ,  7025.49885,
        7027.5    ,  7029.49908,  7031.5    ,  7033.4993 ,  7035.5    ,
        7037.49947,  7039.5    ,  7041.4996 ,  7043.5    ,  7045.4996 ,
        7047.5    ,  7049.49968,  7051.5    ,  7053.4998 ,  7055.5    ,
        7057.49991,  7059.5    ,  7061.50002,  7063.5    ,  7065.50002,
        7067.5    ,  7069.50003,  7071.5    ,  7073.49998,  7075.5    ,
        7077.5001 ,  7079.5    ,  7081.50025,  7083.5    ,  7085.5004 ,
        7087.5    ,  7089.5005 ,  7091.5    ,  7093.50062,  7095.5    ,
        7097.50074,  7099.5    ,  7101.50101,  7103.5    ,  7105.50136,
        7107.5    ,  7109.50174,  7111.5    ,  7113.50159,  7115.5    ,
        7117.49822,  7119.5    ,  7121.49793,  7123.5    ,  7125.49826,
        7127.5    ,  7129.49864,  7131.5    ,  7133.49907,  7135.5    ,
        7137.49956,  7139.5    ,  7141.50039,  7143.5    ,  7145.5016 ,
        7147.5    ,  7149.50292,  7151.5    ,  7153.50325,  7155.5    ,
        7157.50053,  7159.5    ,  7161.50014,  7163.5    ,  7165.49978,
        7167.5    ,  7169.49758,  7171.5    ,  7173.497  ,  7175.5    ,
        7177.49664,  7179.5    ,  7181.4992 ,  7183.5    ,  7185.50217,
        7187.5    ,  7189.50506,  7191.5    ,  7193.49817,  7195.5    ,
        7197.49636,  7199.5    ,  7201.50033,  7203.5    ,  7205.50395,
        7207.5    ,  7209.50214,  7211.5    ,  7213.49972,  7215.5    ,
        7217.49628,  7219.5    ,  7221.49717,  7223.5    ,  7225.50265,
        7227.5    ,  7229.50231,  7231.5    ,  7233.49567,  7235.5    ,
        7237.50425,  7239.5    ,  7241.50262,  7243.5    ,  7245.49822,
        7247.5    ,  7249.49508,  7251.5    ,  7253.50469,  7255.5    ,
        7257.49695,  7259.5    ,  7261.50189,  7263.5    ,  7265.5018 ,
        7267.5    ,  7269.49597,  7271.5    ,  7273.50233,  7275.5    ,
        7277.49708,  7279.5    ,  7281.50218,  7283.5    ,  7285.49925,
        7287.5    ,  7289.49943,  7291.5    ,  7293.4993 ,  7295.5    ,
        7297.50436,  7299.5    ,  7301.49452,  7303.5    ,  7305.50441,
        7307.50001,  7309.49742,  7311.49998,  7313.49835,  7315.49998,
        7317.50151,  7319.50003,  7321.50128,  7323.5    ,  7325.49874,
        7327.49997,  7329.49848,  7331.50002,  7333.50155,  7335.50002,
        7337.50269,  7339.49999,  7341.49555,  7343.5    ,  7345.50542,
        7347.5    ,  7349.4958 ,  7351.5    ,  7353.50053,  7355.5    ,
        7357.5008 ,  7359.5    ,  7361.50054,  7363.5    ,  7365.49793,
        7367.5    ,  7369.50283,  7371.5    ,  7373.49772,  7375.5    ,
        7377.50402,  7379.5    ,  7381.49822,  7383.5    ,  7385.49811,
        7387.5    ,  7389.50313,  7391.5    ,  7393.49521,  7395.5    ,
        7397.50488,  7399.5    ,  7401.5018 ,  7403.5    ,  7405.49746,
        7407.5    ,  7409.49568,  7411.5    ,  7413.5043 ,  7415.5    ,
        7417.49781,  7419.5    ,  7421.49724,  7423.5    ,  7425.5028 ,
        7427.5    ,  7429.50374,  7431.5    ,  7433.50035,  7435.5    ,
        7437.4979 ,  7439.5    ,  7441.49601,  7443.5    ,  7445.49961,
        7447.5    ,  7449.50359,  7451.5    ,  7453.50194,  7455.5    ,
        7457.49496,  7459.5    ,  7461.49779,  7463.5    ,  7465.50076,
        7467.5    ,  7469.50333,  7471.5    ,  7473.50302,  7475.5    ,
        7477.50245,  7479.5    ,  7481.50023,  7483.5    ,  7485.49986,
        7487.5    ,  7489.49948,  7491.5    ,  7493.49678,  7495.5    ,
        7497.49706,  7499.5    ,  7501.49839,  7503.5    ,  7505.49959,
        7507.5    ,  7509.50043,  7511.5    ,  7513.50092,  7515.5    ,
        7517.50136,  7519.5    ,  7521.50174,  7523.5    ,  7525.50207,
        7527.5    ,  7529.5018 ,  7531.5    ,  7533.49845,  7535.5    ,
        7537.49825,  7539.5    ,  7541.49863,  7543.5    ,  7545.49898,
        7547.5    ,  7549.49925,  7551.5    ,  7553.49938,  7555.5    ,
        7557.4995 ,  7559.5    ,  7561.4996 ,  7563.5    ,  7565.49975,
        7567.5    ,  7569.49989,  7571.5    ,  7573.50002,  7575.5    ,
        7577.49998,  7579.5    ,  7583.5    ,  7585.49998,
        7587.5    ,  7589.50008,  7591.5    ,  7593.5002 ,  7595.5    ,
        7597.50032,  7599.5    ,  7601.5004 ,  7603.5    ,  7605.5004 ,
        7607.5    ,  7609.50053,  7611.5    ,  7613.5007 ,  7615.5    ,
        7617.50091,  7619.5    ,  7621.50114,  7623.5    ,  7625.50138,
        7627.5    ,  7629.5016 ,  7631.5    ,  7633.50052,  7635.5    ,
        7637.49828,  7639.5    ,  7641.4984 ,  7643.5    ,  7645.4986 ,
        7647.5    ,  7649.49885,  7651.5    ,  7653.49913,  7655.5    ,
        7657.49943,  7659.5    ,  7661.49975,  7663.5    ,  7665.50009,
        7667.5    ,  7669.50045,  7671.5    ,  7673.50085,  7675.5    ,
        7677.50129,  7679.5    ,  7681.50223,  7683.5    ,  7685.50284,
        7687.5    ,  7689.50051,  7691.5    ,  7693.5    ,  7695.5    ,
        7697.5    ,  7699.5    ,  7701.5    ,  7703.5    ,  7705.49964,
        7707.5    ,  7709.49739,  7711.5    ,  7713.49742,  7715.5    ,
        7717.49777,  7719.5    ,  7721.49867,  7723.5    ,  7725.4996 ,
        7727.5    ,  7729.50054,  7731.5    ,  7733.50144,  7735.5    ,
        7737.50252,  7739.5    ,  7741.50352,  7743.5    ,  7745.49856,
        7747.5    ,  7749.49693,  7751.5    ,  7753.49817,  7755.5    ,
        7757.49945,  7759.5    ,  7761.50076,  7763.5    ,  7765.5021 ,
        7767.5    ,  7769.5034 ,  7771.5    ,  7773.50171,  7775.5    ,
        7777.50037,  7779.5    ,  7781.50005,  7783.5    ,  7785.49895,
        7787.5    ,  7789.49756,  7791.5    ,  7793.49677,  7795.5    ,
        7797.49762,  7799.5    ,  7801.49946,  7803.5    ,  7805.50133,
        7807.5    ,  7809.50322,  7811.5    ,  7813.50388,  7815.5    ,
        7817.4965 ,  7819.5    ,  7821.49704,  7823.5    ,  7825.49892,
        7827.5    ,  7829.50087,  7831.5    ,  7833.50313,  7835.5    ,
        7837.50358,  7839.5    ,  7841.50085,  7843.5    ,  7845.50032,
        7847.5    ,  7849.49943,  7851.5    ,  7853.49562,  7855.5    ,
        7857.49669,  7859.5    ,  7861.4987 ,  7863.5    ,  7865.50111,
        7867.5    ,  7869.50355,  7871.5    ,  7873.50316,  7875.5    ,
        7877.49585,  7879.5    ,  7881.49736,  7883.5    ,  7885.50006,
        7887.5    ,  7889.50247,  7891.5    ,  7893.50354,  7895.5    ,
        7897.50247,  7899.5    ,  7901.49976,  7903.5    ,  7905.49729,
        7907.5    ,  7909.49696,  7911.5    ,  7913.49707,  7915.5    ,
        7917.49979,  7919.5    ,  7921.50274,  7923.5    ,  7925.50457,
        7927.5    ,  7929.49662,  7931.5    ,  7933.49682,  7935.5    ,
        7937.50004,  7939.5    ,  7941.503  ,  7943.5    ,  7945.50285,
        7947.5    ,  7949.50241,  7951.5    ,  7953.49998,  7955.5    ,
        7957.49608,  7959.5    ,  7961.49608,  7963.5    ,  7965.49862,
        7967.5    ,  7969.50199,  7971.5    ,  7973.50491,  7975.5    ,
        7977.49771,  7979.5    ,  7981.49647,  7983.5    ,  7985.50007,
        7987.5    ,  7989.5027 ,  7991.5    ,  7993.50403,  7995.5    ,
        7997.50143,  7999.5    ,  8001.49864,  8003.5    ,  8005.49515,
        8007.5    ,  8009.49724,  8011.5    ,  8013.50042,  8015.5    ,
        8017.50444,  8019.5    ,  8021.50103,  8023.5    ,  8025.49545,
        8027.5    ,  8029.49947,  8031.5    ,  8033.50314,  8035.5    ,
        8037.50267,  8039.5    ,  8041.50162,  8043.5    ,  8045.49858,
        8047.5    ,  8049.49594,  8051.5    ,  8053.49711,  8055.5    ,
        8057.50169,  8059.5    ,  8061.50509,  8063.5    ,  8065.49672,
        8067.5    ,  8069.49747,  8071.5    ,  8073.50176,  8075.5    ,
        8077.50312,  8079.5    ,  8081.50315,  8083.5    ,  8085.49996,
        8087.5    ,  8089.49516,  8091.5    ,  8093.49632,  8095.5    ,
        8097.5013 ,  8099.5    ,  8101.50508,  8103.5    ,  8105.49629,
        8107.5    ,  8109.49806,  8111.5    ,  8113.50262,  8115.5    ,
        8117.50283,  8119.5    ,  8121.5014 ,  8123.5    ,  8125.49777,
        8127.50002,  8129.49586,  8131.49994,  8133.50021,  8135.4992 ,
        8137.50864, 10085.60056, 10915.98877]