﻿
final class ColorTemperature {
  // 360-830[nm](1[nm]刻み)
  private static final int CIX1964_10deg_XYZ_CMF_N = 830 - 360 + 1;

  private static final double[] CIX1964_10deg_XYZ_CMF_X =
  {
0.000000122200, 0.000000185138, 0.000000278830, 0.000000417470, 0.000000621330,
0.000000919270, 0.000001351980, 0.000001976540, 0.000002872500, 0.000004149500,
0.000005958600, 0.000008505600, 0.000012068600, 0.000017022600, 0.000023868000,
0.000033266000, 0.000046087000, 0.000063472000, 0.000086892000, 0.000118246000,
0.000159952000, 0.000215080000, 0.000287490000, 0.000381990000, 0.000504550000,
0.000662440000, 0.000864500000, 0.001121500000, 0.001446160000, 0.001853590000,
0.002361600000, 0.002990600000, 0.003764500000, 0.004710200000, 0.005858100000,
0.007242300000, 0.008899600000, 0.010870900000, 0.013198900000, 0.015929200000,
0.019109700000, 0.022788000000, 0.027011000000, 0.031829000000, 0.037278000000,
0.043400000000, 0.050223000000, 0.057764000000, 0.066038000000, 0.075033000000,
0.084736000000, 0.095041000000, 0.105836000000, 0.117066000000, 0.128682000000,
0.140638000000, 0.152893000000, 0.165416000000, 0.178191000000, 0.191214000000,
0.204492000000, 0.217650000000, 0.230267000000, 0.242311000000, 0.253793000000,
0.264737000000, 0.275195000000, 0.285301000000, 0.295143000000, 0.304869000000,
0.314679000000, 0.324355000000, 0.333570000000, 0.342243000000, 0.350312000000,
0.357719000000, 0.364482000000, 0.370493000000, 0.375727000000, 0.380158000000,
0.383734000000, 0.386327000000, 0.387858000000, 0.388396000000, 0.387978000000,
0.386726000000, 0.384696000000, 0.382006000000, 0.378709000000, 0.374915000000,
0.370702000000, 0.366089000000, 0.361045000000, 0.355518000000, 0.349486000000,
0.342957000000, 0.335893000000, 0.328284000000, 0.320150000000, 0.311475000000,
0.302273000000, 0.292858000000, 0.283502000000, 0.274044000000, 0.264263000000,
0.254085000000, 0.243392000000, 0.232187000000, 0.220488000000, 0.208198000000,
0.195618000000, 0.183034000000, 0.170222000000, 0.157348000000, 0.144650000000,
0.132349000000, 0.120584000000, 0.109456000000, 0.099042000000, 0.089388000000,
0.080507000000, 0.072034000000, 0.063710000000, 0.055694000000, 0.048117000000,
0.041072000000, 0.034642000000, 0.028896000000, 0.023876000000, 0.019628000000,
0.016172000000, 0.013300000000, 0.010759000000, 0.008542000000, 0.006661000000,
0.005132000000, 0.003982000000, 0.003239000000, 0.002934000000, 0.003114000000,
0.003816000000, 0.005095000000, 0.006936000000, 0.009299000000, 0.012147000000,
0.015444000000, 0.019156000000, 0.023250000000, 0.027690000000, 0.032444000000,
0.037465000000, 0.042956000000, 0.049114000000, 0.055920000000, 0.063349000000,
0.071358000000, 0.079901000000, 0.088909000000, 0.098293000000, 0.107949000000,
0.117749000000, 0.127839000000, 0.138450000000, 0.149516000000, 0.161041000000,
0.172953000000, 0.185209000000, 0.197755000000, 0.210538000000, 0.223460000000,
0.236491000000, 0.249633000000, 0.262972000000, 0.276515000000, 0.290269000000,
0.304213000000, 0.318361000000, 0.332705000000, 0.347232000000, 0.361926000000,
0.376772000000, 0.391683000000, 0.406594000000, 0.421539000000, 0.436517000000,
0.451584000000, 0.466782000000, 0.482147000000, 0.497738000000, 0.513606000000,
0.529826000000, 0.546440000000, 0.563426000000, 0.580726000000, 0.598290000000,
0.616053000000, 0.633948000000, 0.651901000000, 0.669824000000, 0.687632000000,
0.705224000000, 0.722773000000, 0.740483000000, 0.758273000000, 0.776083000000,
0.793832000000, 0.811436000000, 0.828822000000, 0.845879000000, 0.862525000000,
0.878655000000, 0.894208000000, 0.909206000000, 0.923672000000, 0.937638000000,
0.951162000000, 0.964283000000, 0.977068000000, 0.989590000000, 1.001910000000,
1.014160000000, 1.026500000000, 1.038800000000, 1.051000000000, 1.062900000000,
1.074300000000, 1.085200000000, 1.095200000000, 1.104200000000, 1.112000000000,
1.118520000000, 1.123800000000, 1.128000000000, 1.131100000000, 1.133200000000,
1.134300000000, 1.134300000000, 1.133300000000, 1.131200000000, 1.128100000000,
1.123990000000, 1.118900000000, 1.112900000000, 1.105900000000, 1.098000000000,
1.089100000000, 1.079200000000, 1.068400000000, 1.056700000000, 1.044000000000,
1.030480000000, 1.016000000000, 1.000800000000, 0.984790000000, 0.968080000000,
0.950740000000, 0.932800000000, 0.914340000000, 0.895390000000, 0.876030000000,
0.856297000000, 0.836350000000, 0.816290000000, 0.796050000000, 0.775610000000,
0.754930000000, 0.733990000000, 0.712780000000, 0.691290000000, 0.669520000000,
0.647467000000, 0.625110000000, 0.602520000000, 0.579890000000, 0.557370000000,
0.535110000000, 0.513240000000, 0.491860000000, 0.471080000000, 0.450960000000,
0.431567000000, 0.412870000000, 0.394750000000, 0.377210000000, 0.360190000000,
0.343690000000, 0.327690000000, 0.312170000000, 0.297110000000, 0.282500000000,
0.268329000000, 0.254590000000, 0.241300000000, 0.228480000000, 0.216140000000,
0.204300000000, 0.192950000000, 0.182110000000, 0.171770000000, 0.161920000000,
0.152568000000, 0.143670000000, 0.135200000000, 0.127130000000, 0.119480000000,
0.112210000000, 0.105310000000, 0.098786000000, 0.092610000000, 0.086773000000,
0.081260600000, 0.076048000000, 0.071114000000, 0.066454000000, 0.062062000000,
0.057930000000, 0.054050000000, 0.050412000000, 0.047006000000, 0.043823000000,
0.040850800000, 0.038072000000, 0.035468000000, 0.033031000000, 0.030753000000,
0.028623000000, 0.026635000000, 0.024781000000, 0.023052000000, 0.021441000000,
0.019941300000, 0.018544000000, 0.017241000000, 0.016027000000, 0.014896000000,
0.013842000000, 0.012862000000, 0.011949000000, 0.011100000000, 0.010311000000,
0.009576880000, 0.008894000000, 0.008258100000, 0.007666400000, 0.007116300000,
0.006605200000, 0.006130600000, 0.005690300000, 0.005281900000, 0.004903300000,
0.004552630000, 0.004227500000, 0.003925800000, 0.003645700000, 0.003385900000,
0.003144700000, 0.002920800000, 0.002713000000, 0.002520200000, 0.002341100000,
0.002174960000, 0.002020600000, 0.001877300000, 0.001744100000, 0.001620500000,
0.001505700000, 0.001399200000, 0.001300400000, 0.001208700000, 0.001123600000,
0.001044760000, 0.000971560000, 0.000903600000, 0.000840480000, 0.000781870000,
0.000727450000, 0.000676900000, 0.000629960000, 0.000586370000, 0.000545870000,
0.000508258000, 0.000473300000, 0.000440800000, 0.000410580000, 0.000382490000,
0.000356380000, 0.000332110000, 0.000309550000, 0.000288580000, 0.000269090000,
0.000250969000, 0.000234130000, 0.000218470000, 0.000203910000, 0.000190350000,
0.000177730000, 0.000165970000, 0.000155020000, 0.000144800000, 0.000135280000,
0.000126390000, 0.000118100000, 0.000110370000, 0.000103150000, 0.000096427000,
0.000090151000, 0.000084294000, 0.000078830000, 0.000073729000, 0.000068969000,
0.000064525800, 0.000060376000, 0.000056500000, 0.000052880000, 0.000049498000,
0.000046339000, 0.000043389000, 0.000040634000, 0.000038060000, 0.000035657000,
0.000033411700, 0.000031315000, 0.000029355000, 0.000027524000, 0.000025811000,
0.000024209000, 0.000022711000, 0.000021308000, 0.000019994000, 0.000018764000,
0.000017611500, 0.000016532000, 0.000015521000, 0.000014574000, 0.000013686000,
0.000012855000, 0.000012075000, 0.000011345000, 0.000010659000, 0.000010017000,
0.000009413630, 0.000008847900, 0.000008317100, 0.000007819000, 0.000007351600,
0.000006913000, 0.000006501500, 0.000006115300, 0.000005752900, 0.000005412700,
0.000005093470, 0.000004793800, 0.000004512500, 0.000004248300, 0.000004000200,
0.000003767100, 0.000003548000, 0.000003342100, 0.000003148500, 0.000002966500,
0.000002795310, 0.000002634500, 0.000002483400, 0.000002341400, 0.000002207800,
0.000002082000, 0.000001963600, 0.000001851900, 0.000001746500, 0.000001647100,
0.000001553140,
  };

  private static final double[] CIX1964_10deg_XYZ_CMF_Y =
  {
0.000000013398, 0.000000020294, 0.000000030560, 0.000000045740, 0.000000068050,
0.000000100650, 0.000000147980, 0.000000216270, 0.000000314200, 0.000000453700,
0.000000651100, 0.000000928800, 0.000001317500, 0.000001857200, 0.000002602000,
0.000003625000, 0.000005019000, 0.000006907000, 0.000009449000, 0.000012848000,
0.000017364000, 0.000023327000, 0.000031150000, 0.000041350000, 0.000054560000,
0.000071560000, 0.000093300000, 0.000120870000, 0.000155640000, 0.000199200000,
0.000253400000, 0.000320200000, 0.000402400000, 0.000502300000, 0.000623200000,
0.000768500000, 0.000941700000, 0.001147800000, 0.001390300000, 0.001674000000,
0.002004400000, 0.002386000000, 0.002822000000, 0.003319000000, 0.003880000000,
0.004509000000, 0.005209000000, 0.005985000000, 0.006833000000, 0.007757000000,
0.008756000000, 0.009816000000, 0.010918000000, 0.012058000000, 0.013237000000,
0.014456000000, 0.015717000000, 0.017025000000, 0.018399000000, 0.019848000000,
0.021391000000, 0.022992000000, 0.024598000000, 0.026213000000, 0.027841000000,
0.029497000000, 0.031195000000, 0.032927000000, 0.034738000000, 0.036654000000,
0.038676000000, 0.040792000000, 0.042946000000, 0.045114000000, 0.047333000000,
0.049602000000, 0.051934000000, 0.054337000000, 0.056822000000, 0.059399000000,
0.062077000000, 0.064737000000, 0.067285000000, 0.069764000000, 0.072218000000,
0.074704000000, 0.077272000000, 0.079979000000, 0.082874000000, 0.086000000000,
0.089456000000, 0.092947000000, 0.096275000000, 0.099535000000, 0.102829000000,
0.106256000000, 0.109901000000, 0.113835000000, 0.118167000000, 0.122932000000,
0.128201000000, 0.133457000000, 0.138323000000, 0.143042000000, 0.147787000000,
0.152761000000, 0.158102000000, 0.163941000000, 0.170362000000, 0.177425000000,
0.185190000000, 0.193025000000, 0.200313000000, 0.207156000000, 0.213644000000,
0.219940000000, 0.226170000000, 0.232467000000, 0.239025000000, 0.245997000000,
0.253589000000, 0.261876000000, 0.270643000000, 0.279645000000, 0.288694000000,
0.297665000000, 0.306469000000, 0.315035000000, 0.323335000000, 0.331366000000,
0.339133000000, 0.347860000000, 0.358326000000, 0.370001000000, 0.382464000000,
0.395379000000, 0.408482000000, 0.421588000000, 0.434619000000, 0.447601000000,
0.460777000000, 0.474340000000, 0.488200000000, 0.502340000000, 0.516740000000,
0.531360000000, 0.546190000000, 0.561180000000, 0.576290000000, 0.591500000000,
0.606741000000, 0.622150000000, 0.637830000000, 0.653710000000, 0.669680000000,
0.685660000000, 0.701550000000, 0.717230000000, 0.732570000000, 0.747460000000,
0.761757000000, 0.775340000000, 0.788220000000, 0.800460000000, 0.812140000000,
0.823330000000, 0.834120000000, 0.844600000000, 0.854870000000, 0.865040000000,
0.875211000000, 0.885370000000, 0.895370000000, 0.905150000000, 0.914650000000,
0.923810000000, 0.932550000000, 0.940810000000, 0.948520000000, 0.955600000000,
0.961988000000, 0.967540000000, 0.972230000000, 0.976170000000, 0.979460000000,
0.982200000000, 0.984520000000, 0.986520000000, 0.988320000000, 0.990020000000,
0.991761000000, 0.993530000000, 0.995230000000, 0.996770000000, 0.998090000000,
0.999110000000, 0.999770000000, 1.000000000000, 0.999710000000, 0.998850000000,
0.997340000000, 0.995260000000, 0.992740000000, 0.989750000000, 0.986300000000,
0.982380000000, 0.977980000000, 0.973110000000, 0.967740000000, 0.961890000000,
0.955552000000, 0.948601000000, 0.940981000000, 0.932798000000, 0.924158000000,
0.915175000000, 0.905954000000, 0.896608000000, 0.887249000000, 0.877986000000,
0.868934000000, 0.860164000000, 0.851519000000, 0.842963000000, 0.834393000000,
0.825623000000, 0.816764000000, 0.807544000000, 0.797947000000, 0.787893000000,
0.777405000000, 0.766490000000, 0.755309000000, 0.743845000000, 0.732190000000,
0.720353000000, 0.708281000000, 0.696055000000, 0.683621000000, 0.671048000000,
0.658341000000, 0.645545000000, 0.632718000000, 0.619815000000, 0.606887000000,
0.593878000000, 0.580781000000, 0.567653000000, 0.554490000000, 0.541228000000,
0.527963000000, 0.514634000000, 0.501363000000, 0.488124000000, 0.474935000000,
0.461834000000, 0.448823000000, 0.435917000000, 0.423153000000, 0.410526000000,
0.398057000000, 0.385835000000, 0.373951000000, 0.362311000000, 0.350863000000,
0.339554000000, 0.328309000000, 0.317118000000, 0.305936000000, 0.294737000000,
0.283493000000, 0.272222000000, 0.260990000000, 0.249877000000, 0.238946000000,
0.228254000000, 0.217853000000, 0.207780000000, 0.198072000000, 0.188748000000,
0.179828000000, 0.171285000000, 0.163059000000, 0.155151000000, 0.147535000000,
0.140211000000, 0.133170000000, 0.126400000000, 0.119892000000, 0.113640000000,
0.107633000000, 0.101870000000, 0.096347000000, 0.091063000000, 0.086010000000,
0.081187000000, 0.076583000000, 0.072198000000, 0.068024000000, 0.064052000000,
0.060281000000, 0.056697000000, 0.053292000000, 0.050059000000, 0.046998000000,
0.044096000000, 0.041345000000, 0.038750700000, 0.036297800000, 0.033983200000,
0.031800400000, 0.029739500000, 0.027791800000, 0.025955100000, 0.024226300000,
0.022601700000, 0.021077900000, 0.019650500000, 0.018315300000, 0.017068600000,
0.015905100000, 0.014818300000, 0.013800800000, 0.012849500000, 0.011960700000,
0.011130300000, 0.010355500000, 0.009633200000, 0.008959900000, 0.008332400000,
0.007748800000, 0.007204600000, 0.006697500000, 0.006225100000, 0.005785000000,
0.005375100000, 0.004994100000, 0.004639200000, 0.004309300000, 0.004002800000,
0.003717740000, 0.003452620000, 0.003205830000, 0.002976230000, 0.002762810000,
0.002564560000, 0.002380480000, 0.002209710000, 0.002051320000, 0.001904490000,
0.001768470000, 0.001642360000, 0.001525350000, 0.001416720000, 0.001315950000,
0.001222390000, 0.001135550000, 0.001054940000, 0.000980140000, 0.000910660000,
0.000846190000, 0.000786290000, 0.000730680000, 0.000678990000, 0.000631010000,
0.000586440000, 0.000545110000, 0.000506720000, 0.000471110000, 0.000438050000,
0.000407410000, 0.000378962000, 0.000352543000, 0.000328001000, 0.000305208000,
0.000284041000, 0.000264375000, 0.000246109000, 0.000229143000, 0.000213376000,
0.000198730000, 0.000185115000, 0.000172454000, 0.000160678000, 0.000149730000,
0.000139550000, 0.000130086000, 0.000121290000, 0.000113106000, 0.000105501000,
0.000098428000, 0.000091853000, 0.000085738000, 0.000080048000, 0.000074751000,
0.000069819000, 0.000065222000, 0.000060939000, 0.000056942000, 0.000053217000,
0.000049737000, 0.000046491000, 0.000043464000, 0.000040635000, 0.000038000000,
0.000035540500, 0.000033244800, 0.000031100600, 0.000029099000, 0.000027230700,
0.000025486000, 0.000023856100, 0.000022333200, 0.000020910400, 0.000019580800,
0.000018338400, 0.000017177700, 0.000016093400, 0.000015080000, 0.000014133600,
0.000013249000, 0.000012422600, 0.000011649900, 0.000010927700, 0.000010251900,
0.000009619600, 0.000009028100, 0.000008474000, 0.000007954800, 0.000007468600,
0.000007012800, 0.000006585800, 0.000006185700, 0.000005810700, 0.000005459000,
0.000005129800, 0.000004820600, 0.000004531200, 0.000004259100, 0.000004004200,
0.000003764730, 0.000003539950, 0.000003329140, 0.000003131150, 0.000002945290,
0.000002770810, 0.000002607050, 0.000002453290, 0.000002308940, 0.000002173380,
0.000002046130, 0.000001926620, 0.000001814400, 0.000001708950, 0.000001609880,
0.000001516770, 0.000001429210, 0.000001346860, 0.000001269450, 0.000001196620,
0.000001128090, 0.000001063680, 0.000001003130, 0.000000946220, 0.000000892630,
0.000000842160, 0.000000794640, 0.000000749780, 0.000000707440, 0.000000667480,
0.000000629700,
  };

  private static final double[] CIX1964_10deg_XYZ_CMF_Z =
  {
0.000000535027, 0.000000810720, 0.000001221200, 0.000001828700, 0.000002722200,
0.000004028300, 0.000005925700, 0.000008665100, 0.000012596000, 0.000018201000,
0.000026143700, 0.000037330000, 0.000052987000, 0.000074764000, 0.000104870000,
0.000146220000, 0.000202660000, 0.000279230000, 0.000382450000, 0.000520720000,
0.000704776000, 0.000948230000, 0.001268200000, 0.001686100000, 0.002228500000,
0.002927800000, 0.003823700000, 0.004964200000, 0.006406700000, 0.008219300000,
0.010482200000, 0.013289000000, 0.016747000000, 0.020980000000, 0.026127000000,
0.032344000000, 0.039802000000, 0.048691000000, 0.059210000000, 0.071576000000,
0.086010900000, 0.102740000000, 0.122000000000, 0.144020000000, 0.168990000000,
0.197120000000, 0.228570000000, 0.263470000000, 0.301900000000, 0.343870000000,
0.389366000000, 0.437970000000, 0.489220000000, 0.542900000000, 0.598810000000,
0.656760000000, 0.716580000000, 0.778120000000, 0.841310000000, 0.906110000000,
0.972542000000, 1.038900000000, 1.103100000000, 1.165100000000, 1.224900000000,
1.282500000000, 1.338200000000, 1.392600000000, 1.446100000000, 1.499400000000,
1.553480000000, 1.607200000000, 1.658900000000, 1.708200000000, 1.754800000000,
1.798500000000, 1.839200000000, 1.876600000000, 1.910500000000, 1.940800000000,
1.967280000000, 1.989100000000, 2.005700000000, 2.017400000000, 2.024400000000,
2.027300000000, 2.026400000000, 2.022300000000, 2.015300000000, 2.006000000000,
1.994800000000, 1.981400000000, 1.965300000000, 1.946400000000, 1.924800000000,
1.900700000000, 1.874100000000, 1.845100000000, 1.813900000000, 1.780600000000,
1.745370000000, 1.709100000000, 1.672300000000, 1.634700000000, 1.595600000000,
1.554900000000, 1.512200000000, 1.467300000000, 1.419900000000, 1.370000000000,
1.317560000000, 1.262400000000, 1.205000000000, 1.146600000000, 1.088000000000,
1.030200000000, 0.973830000000, 0.919430000000, 0.867460000000, 0.818280000000,
0.772125000000, 0.728290000000, 0.686040000000, 0.645530000000, 0.606850000000,
0.570060000000, 0.535220000000, 0.502340000000, 0.471400000000, 0.442390000000,
0.415254000000, 0.390024000000, 0.366399000000, 0.344015000000, 0.322689000000,
0.302356000000, 0.283036000000, 0.264816000000, 0.247848000000, 0.232318000000,
0.218502000000, 0.205851000000, 0.193596000000, 0.181736000000, 0.170281000000,
0.159249000000, 0.148673000000, 0.138609000000, 0.129096000000, 0.120215000000,
0.112044000000, 0.104710000000, 0.098196000000, 0.092361000000, 0.087088000000,
0.082248000000, 0.077744000000, 0.073456000000, 0.069268000000, 0.065060000000,
0.060709000000, 0.056457000000, 0.052609000000, 0.049122000000, 0.045954000000,
0.043050000000, 0.040368000000, 0.037839000000, 0.035384000000, 0.032949000000,
0.030451000000, 0.028029000000, 0.025862000000, 0.023920000000, 0.022174000000,
0.020584000000, 0.019127000000, 0.017740000000, 0.016403000000, 0.015064000000,
0.013676000000, 0.012308000000, 0.011056000000, 0.009915000000, 0.008872000000,
0.007918000000, 0.007030000000, 0.006223000000, 0.005453000000, 0.004714000000,
0.003988000000, 0.003289000000, 0.002646000000, 0.002063000000, 0.001533000000,
0.001091000000, 0.000711000000, 0.000407000000, 0.000184000000, 0.000047000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
0.000000000000,
  };

  private static final char[] HEX_CHARACTER = {
    '0','1','2','3','4','5','6','7',
    '8','9','a','b','c','d','e','f',
  };

  public static void main(String[] args) {
    System.out.println("<html><body bgcolor=black>"
      + "<table width=100% align=center hspace=0 vspace=0 border=0>"
      + "<caption><font color=white>色温度表</font></caption>"
      + "<tr align=center>"
      + "<th><font color=white>温度 [K]</font></th>"
      + "<th><font color=white>R</font></th>"
      + "<th><font color=white>G</font></th>"
      + "<th><font color=white>B</font></tr>");
    for (int temperature = 0; temperature < 26000; temperature += 10) {
      double X = 0;
      double Y = 0;
      double Z = 0;

      for (int j = 0; j < CIX1964_10deg_XYZ_CMF_N; ++j) {
        final double lambda = (360 + j) * 1e-9; // 波長 [m]
        final double spd = findBlackBodyRadiance(lambda, temperature);

        X += spd * CIX1964_10deg_XYZ_CMF_X[j];
        Y += spd * CIX1964_10deg_XYZ_CMF_Y[j];
        Z += spd * CIX1964_10deg_XYZ_CMF_Z[j];
      }

      final double x = X / (X + Y + Z);
      final double y = Y / (X + Y + Z);

      Y = 1.0;
      X = Y / y * x;
      Z = Y / y * (1.0 - x - y);

      final double R =  3.240479 * X - 1.537150 * Y - 0.498535 * Z;
      final double G = -0.969256 * X + 1.875991 * Y + 0.041556 * Z;
      final double B =  0.055648 * X - 0.204043 * Y + 1.057311 * Z;

      final int r = Math.min(Math.max((int)(R * 255), 0), 255);
      final int g = Math.min(Math.max((int)(G * 255), 0), 255);
      final int b = Math.min(Math.max((int)(B * 255), 0), 255);

      System.out.printf( "<tr bgcolor=\"#%c%c%c%c%c%c\" align=right>"
              + "<td><font size=1>%d</font></td>"
              + "<td><font size=1>%d</font></td>"
              + "<td><font size=1>%d</font></td>"
              + "<td><font size=1>%d</font></td></tr>\n",
          HEX_CHARACTER[r>>4], HEX_CHARACTER[r&15],
          HEX_CHARACTER[g>>4], HEX_CHARACTER[g&15],
          HEX_CHARACTER[b>>4], HEX_CHARACTER[b&15],
          temperature, r, g, b);
    }
    System.out.println("</table></body></html>");
  }

  /**
   * プランクの法則に基づいて黒体の分光放射輝度を求める.
   * cf. http://ja.wikipedia.org/wiki/%E9%BB%92%E4%BD%93
   * cf. http://ja.wikipedia.org/wiki/%E3%83%97%E3%83%A9%E3%83%B3%E3%82%AF%E3%81%AE%E6%B3%95%E5%89%87
   *
   * I(lambda, T) = 2 h c^2 / (lambda^5 (exp(h c / (lambda k T)) - 1))
   *
   *   lambda: 波長           [m]
   *   T     : 温度           [K]
   *   h     : プランク定数   6.62606896e-34 [J.s]
   *   c     : 光速度         2.99792458e+8  [m/s]
   *   k     : ボルツマン定数 1.38065030e-23 [J/K]
   *
   * @param lambda 波長 [m]
   * @param t      温度 [K]
   */
  private static double findBlackBodyRadiance(final double lambda, final double temperature) {
    final double lambda5 = lambda * lambda * lambda * lambda * lambda;
    final double c1 = (2.0 * 6.62606896e-18 * 2.99792458 * 2.99792458) / lambda5;
    final double c2 = (6.62606896e-3 * 2.99792458 / 1.38065030) / (lambda * temperature);
    return c1 / (Math.exp(c2) - 1);
  }
}
