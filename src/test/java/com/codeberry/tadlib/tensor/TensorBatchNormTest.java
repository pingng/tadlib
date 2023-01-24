package com.codeberry.tadlib.tensor;

import com.codeberry.tadlib.provider.ProviderStore;
import com.codeberry.tadlib.provider.java.NDArray;
import com.codeberry.tadlib.provider.java.Shape;
import org.junit.jupiter.api.Test;

import static com.codeberry.tadlib.array.TArrayFactory.*;
import static com.codeberry.tadlib.tensor.Ops.RunMode;
import static com.codeberry.tadlib.util.MatrixTestUtils.assertEqualsMatrix;
import static java.lang.Math.toIntExact;
import static org.junit.jupiter.api.Assertions.assertNull;

public class TensorBatchNormTest {

    @Test
    public void averageUpdate() {

        OpsExtended.BatchNormResult result = new OpsExtended.BatchNormResult(null,
                ProviderStore.array(new double[]{1, 2}),
                ProviderStore.array(new double[]{3, 4}));
        OpsExtended.BatchNormRunningAverages averages = new OpsExtended.BatchNormRunningAverages();

        assertNull(averages.running.mean);
        assertNull(averages.running.variance);

        averages.updateWith(result, 0.9);

        assertEqualsMatrix(ProviderStore.array(new double[]{1, 2}).toDoubles(),
                averages.running.mean.toDoubles());
        assertEqualsMatrix(ProviderStore.array(new double[]{3, 4}).toDoubles(),
                averages.running.variance.toDoubles());

        OpsExtended.BatchNormResult nextResult = new OpsExtended.BatchNormResult(null,
                ProviderStore.array(new double[]{50, 60}),
                ProviderStore.array(new double[]{70, 80}));
        averages.updateWith(nextResult, 0.9);
        assertEqualsMatrix(ProviderStore.array(new double[]{
                        1 * 0.9 + 50 * 0.1,
                        2 * 0.9 + 60 * 0.1}).toDoubles(),
                averages.running.mean.toDoubles());
        assertEqualsMatrix(ProviderStore.array(new double[]{
                        3 * 0.9 + 70 * 0.1,
                        4 * 0.9 + 80 * 0.1}).toDoubles(),
                averages.running.variance.toDoubles());

    }

    @Test
    public void batchNorm2DForEachChannel() {
        Tensor input = new Tensor(Data2D.testInput().reshape(8, 16));
        NDArray ndArray = input.val();
        Shape inputShape = ndArray.shape;
        Tensor gamma = new Tensor(onesShaped(1));
        Tensor beta = new Tensor(zerosShaped(1));

        OpsExtended.BatchNormRunningAverages averages = new OpsExtended.BatchNormRunningAverages();
        OpsExtended.BatchNormResult batchNormResult = OpsExtended.batchNorm(input, beta, gamma, averages, RunMode.TRAINING);
        batchNormResult.output.backward(range(toIntExact(inputShape.size)).reshape(inputShape.toDimArray()));

        assertEqualsMatrix(Data2D.testOutput().reshape(inputShape.toDimArray()).toDoubles(),
                batchNormResult.output.val().toDoubles());
        assertEqualsMatrix(Data2D.meanValues().toDoubles(),
                batchNormResult.mean.toDoubles());
        assertEqualsMatrix(Data2D.varianceValues().toDoubles(),
                batchNormResult.variance.toDoubles());
        assertEqualsMatrix(Data2D.inputGradient().reshape(inputShape.toDimArray()).toDoubles(),
                input.grad().toDoubles());
        assertEqualsMatrix(Data2D.betaGradient().toDoubles(),
                beta.grad().toDoubles());
        assertEqualsMatrix(Data2D.gammaGradient().toDoubles(),
                gamma.grad().toDoubles());
    }

    @Test
    public void batchNorm4DForEachChannel() {
        Tensor input = new Tensor(Data4D.testInput().reshape(8, 3, 3, 5));
        NDArray ndArray = input.val();
        Shape inputShape = ndArray.shape;
        Tensor gamma = new Tensor(onesShaped(5));
        Tensor beta = new Tensor(zerosShaped(5));

        OpsExtended.BatchNormRunningAverages averages = new OpsExtended.BatchNormRunningAverages();
        OpsExtended.BatchNormResult batchNormResult = OpsExtended.batchNorm(input, beta, gamma, averages, RunMode.TRAINING);
        batchNormResult.output.backward(range(toIntExact(inputShape.size)).reshape(inputShape.toDimArray()));

        assertEqualsMatrix(Data4D.testOutput().reshape(inputShape.toDimArray()).toDoubles(),
                batchNormResult.output.val().toDoubles(),
                1.8e-7);
        assertEqualsMatrix(Data4D.meanValues().toDoubles(),
                batchNormResult.mean.toDoubles(),
                1.8e-7);
        assertEqualsMatrix(Data4D.varianceValues().toDoubles(),
                batchNormResult.variance.toDoubles(),
                1.8e-7);
        assertEqualsMatrix(Data4D.inputGradient().reshape(inputShape.toDimArray()).toDoubles(),
                input.grad().toDoubles(),
                1.0e-7);
        assertEqualsMatrix(Data4D.betaGradient().toDoubles(),
                beta.grad().toDoubles(),
                3.1e-7);
        assertEqualsMatrix(Data4D.gammaGradient().toDoubles(),
                gamma.grad().toDoubles(),
                3.1e-7);
    }

    private static class Data2D {
        private static NDArray meanValues() {
            return ProviderStore.array(new double[]{
                    1.5038592291504351, -3.2113610045557506, 10.7732774171762689, -13.7949849404274190, 27.5839334672141199, -8.6703345984551579, 213.9156380505953052, 413.2545734874228174, 148.7379536583603965, 225.6029062357371799, 600.2503630947087458, 893.0466265182635652, 320.1612701885728143, 750.1298954938704355, 695.2255327874495379, 1207.8393562995729553
            });
        }

        private static NDArray varianceValues() {
            return ProviderStore.array(new double[]{
                    0.0998707636807110, 0.2651521348779396, 0.8579147893744125, 1.0358379492554346, 0.4391801863357090, 3.4187508013083256, 2.8024127749412582, 7.3142349043806538, 5.8670829662423944, 5.6636933861920689, 10.3786513833000047, 12.0955556500564931, 13.6593014708455769, 16.2615733512566507, 8.2055304525699118, 20.1879750836942655
            });
        }

        private static NDArray inputGradient() {
            return ProviderStore.array(new double[]{
                    -158.1668524289757727, -96.6996262185303408, -71.9531107854016057, -49.0351986506428688, -41.0081464184869020, -24.3358478431438741, -3.1441572703040706, -4.0177809489280527, -14.6152893007208178, -17.8319921466381857, -14.4479998735131492, -9.7374044290980066, -14.9080204363973721, -10.0163535991603130, -24.6967504443053230, -11.0180114915609249, -146.9094788220108967, -78.2911617850365360, -26.7120047638315228, -24.4615300019415685, -79.7421869257251501, -28.5535316596282769, -7.3968692604159827, -20.1633834000314422, -5.5275465873870218, 5.2315022758685750, -11.9641661043403893, -4.2883715510145564, -10.7133622323168112, -11.5749950180898971, -8.9557024502389933, -2.8765169449636687, -75.2103521595305864, -69.1614459080171997, -12.8733756382624378, -23.8026700931958786, -34.3840562102043990, -14.4318739369385440, -25.8143070629985800, -8.6941650019327721, -5.8798309745767057, -12.9738808336034115, -4.6814852246260230, -0.0020266535719458, -6.7555388312824292, -6.7029346018042713, 1.0583533087934143, -8.5224632759312637, -23.2234379214028195, 2.6412882460948595, -17.8058072241195546, -14.7571283603527164, 16.9583073048423074, 2.0571025666317340, -9.2558283670378572, 9.2103568628977612, -15.4916751937207113, 2.4721739186013565, 1.1575337991197259, -2.1924321017234760, -1.8772251968990901, -3.6743790490318169, -3.2724087661305568, -0.6741640848568675, 39.3521311458944183, 52.3221596773434214, 8.4986745494518274, -9.2200965827888197, 31.4169298267767232, 9.7620075283803089, 17.3714402991439627, -3.1426079310938384, 3.6667930261316961, 1.3965419381752646, -4.5092502842073294, -1.1129458398482726, 1.6471009760598534, 2.3092649536794170, 3.2190504579025436, 5.5203919527531191, 57.2314393570637492, 24.1831331429094121, 23.1122321070072516, 22.2413434737006241, 49.4006849904320546, 7.9087929744381142, 9.1285011667010707, 4.9412099377033627, 10.4119840265955901, -6.5548446498865296, 8.8215988407478747, 14.6633035507616611, 6.2205633974097552, 8.6915508353198554, 8.4317941857800989, 1.2901674558519325, 125.2885420037481481, 77.5591732044440505, 42.3470139399553247, 35.1109004794353439, -1.1445969159070160, 24.1355833669652853, 13.6915624939889398, 8.4902662354853398, 2.4207003550379582, 13.9343201550246540, 5.6754999657843683, 2.5787634696165291, 10.8757751607206608, 12.2771628774933461, 13.9383031850112076, 3.0990824459107777, 181.6380088252137170, 87.4464796407924325, 55.3863778152006958, 63.9243797357859052, 58.5030643482734405, 23.4577670032952170, 5.4196580009251605, 13.3761042458990786, 25.0148646486407031, 14.3261793424583992, 19.9482688810359221, 0.0911135548805113, 15.5107071627054331, 8.6906836015937579, 10.2773605231867577, 13.1815139427970855
            });
        }

        private static NDArray betaGradient() {
            return ProviderStore.array(new double[]{
                    8128.0000000000000000
            });
        }

        private static NDArray gammaGradient() {
            return ProviderStore.array(new double[]{
                    -120.1300841775505432
            });
        }

        private static NDArray testOutput() {
            return ProviderStore.array(new double[]{
                    1.4656140986650890, 0.5939163969467876, 1.2359419056129450, 0.6428311191291858, 1.3651734759592387, -1.0995906204528962, 1.7432565091075531, -1.9244819580019907, -1.0160622031683744, -0.5273399521218209, 0.7210253003680407, -0.7689787644116564, 0.8273177769788163, 1.3980469127799324, -0.9662664657378173, 0.3823608626123587, -1.5659306162927860, -0.0300942330737222, -1.7714650457315031, 1.5932970905657200, -0.6084135959136816, 1.2786454572621255, 0.9489037053158853, 0.6196156639965977, -1.3126316699637073, -2.0393792810608886, 0.1110652391121505, -0.8714998565918677, 0.3714090744537515, -0.5980515890377660, 0.9401541172527743, 1.5939516131501819, 0.0564414099948101, -1.1113199493156127, -1.4020240858268593, -0.0237820424241377, 0.0574717127023661, 0.2682529108149980, -0.6601808712183725, -0.0207553850691511, -0.4813188587559409, 0.2673506439311950, 0.6801270368879671, -0.8335395987711252, -0.8871533658642505, -0.2713986927687131, 1.7714855902889610, -0.8413966426165302, 0.1610051204908460, 0.8956994012107282, 0.9859524617397142, -0.7404490460931896, 0.9111892173848446, -1.1795579674884262, -0.2575081550652101, -1.4032287418234830, 1.4563142336712929, -0.5398174564632257, 0.8944976252901142, -0.0130283493534193, 0.9739056214816770, -0.6106145630557762, -0.0900375459800102, 0.2926405149072480, 1.0808780416823298, 1.8126526387638942, 0.0148848810041713, -1.8337971956355918, 0.6072054256821815, -1.0043031430389480, 0.7243048289912650, 0.7035118585335738, -0.0434925442548675, 0.1818301776813769, -1.7179760379518711, 0.4123994301824041, -1.7538166127614829, 0.1175391951132099, 0.0800209554649314, 0.9892433097863886, -1.4407884722638391, -1.1050094429321655, 0.3009943423908155, -0.1438453286280925, 0.4138679653809945, 0.9370411189646304, -0.2995592887754839, 0.4535363142381073, -0.0601769499040046, 1.5397180457133430, 0.3370515353486212, -0.9379038198837861, -0.9259248899593757, 0.9896812776277670, 0.0100349364656154, -1.0716299999346575, -0.0988463753259534, -0.0059748790432490, 0.0901630180727349, -0.4499578595465206, -1.9304485691624933, -0.4623246524716471, -0.5868432853009580, 0.7264975737880093, 1.6838336557877582, 0.2658914179930179, -1.6561185968702432, 1.0780626277276042, 0.1790224599851484, 0.8516726429932930, -0.0048045359427817, -1.5350804408950012, 0.3416267930494969, -1.0498699325566596, 0.5455525227379763, 0.9557032626326301, -0.8160456320334433, 1.2618368964101667, -1.6123734430545795, 0.8453046743385357, -0.2264656634122488, 0.8517464043270309, 0.6303278978154196, 1.9344883311014769, 1.2152399356856733, -1.8768751836518902, -1.7405870518117297, 0.1899107829899549
            });
        }

        private static NDArray testInput() {
            return ProviderStore.array(new double[]{
                    1.9670298390136767, -2.9055355016485551, 11.9180530798946531, -13.1407360253025409, 28.4886441229863543, -10.7034630265177420, 216.8339211834336879, 408.0498420416367367, 146.2768412614510112, 224.3479153240444646, 602.5732121397312540, 890.3722208952030996, 323.2189120627899115, 755.7676094804543254, 692.4576336210704994, 1209.5573431029256426, 1.0089860976675551, -3.2268574347127412, 9.1324801737944981, -12.1733881291430563, 27.1807332343989891, -6.3061361593106238, 215.5041419015454380, 414.9303143886535850, 145.5584887934635390, 220.7494858701310818, 600.6081699351555017, 890.0156662046208567, 321.5339421776374138, 747.7182140421281247, 697.9186323512687977, 1215.0011453432150574, 1.5216961224449581, -3.7836132342255384, 9.4746702215543976, -13.8191893931871483, 27.6220204003423646, -8.1743385352440381, 212.8104681051319176, 413.1984408721547197, 147.5721000261155780, 226.2391611354822203, 602.4414546379913418, 890.1476869204936975, 316.8824849916380231, 749.0354628305771030, 700.3000061987452227, 1204.0588743785822317, 1.5547408629425585, -2.7501383118718046, 11.6865034981684381, -14.5485856462674938, 28.1877853650936849, -10.8513215872862183, 213.4845588023211462, 409.4595632365696360, 152.2654473522578655, 224.3182207134122734, 603.1320691230818056, 893.0013156561403775, 323.7606786536637742, 747.6675530423526652, 694.9676175463667960, 1209.1542203243639051, 1.8454436490941133, -2.2779713764938561, 10.7870643512050037, -15.6613535559183585, 27.9863330365958731, -10.5272777485013869, 215.1281541401297659, 415.1572103385339005, 148.6326057382183023, 226.0356351036020612, 594.7157459362108511, 894.4808967557137294, 313.6794281877157005, 750.6038798489046258, 695.4547552056725408, 1212.2841285492918360, 1.0485341364195526, -3.7803637637643117, 11.0520696453708833, -13.9413852116653736, 27.8582068580737463, -6.9377580360560671, 213.4141634740477116, 414.4811553770091450, 148.5921926636544583, 229.2672069324025301, 601.3362054869359099, 889.7847220732087408, 316.7391912792348307, 754.1208488232174432, 695.2542781628327475, 1203.0244121512985203, 1.4726213091657732, -3.2144376503557401, 10.8567897799322814, -14.2529348322419480, 26.3046107864724767, -9.5251667300234590, 212.9332382418660927, 415.2193748470091350, 152.8165463583643771, 226.2356883940691432, 594.9150253482795279, 896.7959846111518800, 320.8229102322267750, 753.5643201383529686, 695.2117700506657911, 1200.9420815065300303, 1.6118218164552922, -3.7519707633734560, 11.2785885874899954, -12.8223067296934303, 27.0431339337494911, -6.3372149647017260, 211.2164585562869377, 415.5406867978158516, 148.1894070733578417, 227.6299364127536364, 602.2810221502845707, 899.7745190295751172, 324.6526139236760855, 742.5612757449763421, 690.2395691629734529, 1208.6926450403766466
            });
        }
    }

    private static class Data4D {
        private static NDArray meanValues() {
            return ProviderStore.array(new double[]{
                    1.4806420704615204, -3.0564849411568065, 10.6332331397539761, -14.0714765137064468, 27.8379306179145694,
            });
        }

        private static NDArray varianceValues() {
            return ProviderStore.array(new double[]{
                    0.0705736722460237, 0.3502831688554585, 0.6659366301149672, 1.3699496108481859, 2.2650502681197846,
            });
        }

        private static NDArray inputGradient() {
            return ProviderStore.array(new double[]{
                    -584.7192800998020630, -296.6286052765131558, -216.7814767939974274, -144.2623956421773812, -117.9303705516193190, -694.7077995413524150, -269.5383010057992692, -212.3010651442710923, -154.6558709882772007, -114.6266068181736415, -579.2643182792621701, -294.9204525580479412, -204.7142198787916243, -127.1898667956579629, -111.3234060737099753, -591.6702601870665603, -294.6714443351319233, -199.3988816799667063, -152.7432084363177580, -107.9454567947415597, -600.4977744580476156, -245.3800405482863027, -192.5907303039329292, -122.3649607934627852, -104.6782712067340810, -643.6333604470911496, -252.0629685766950843, -187.5171843429098999, -122.3135713383187522, -101.3394987352815093, -547.1154556669122258, -228.9793999903404824, -180.7874270247722279, -137.8942601643321950, -98.0348414120409757, -525.3279012171440172, -238.4837097517697089, -174.4628548136858797, -124.3087062675502779, -94.6780589637142640, -531.9452972421927370, -225.7117270640591755, -168.1146325574218565, -127.1029224993406359, -91.3750887025214240, -495.0048153603825085, -214.5370390254277595, -162.8628525763374455, -110.8987292209202735, -88.0353270907026371, -408.7774282427340609, -220.1588530626255817, -156.0817191482684336, -118.1631329640528634, -84.7220208919593176, -512.2675626163553488, -187.5725875574005101, -150.3051791256768581, -93.6082826882512506, -81.4054739983317432, -369.5628522613425275, -201.4389284097173345, -144.3495890390802288, -97.5306698207665193, -78.0531124305982473, -358.2296235157941737, -184.6976852690659712, -138.6425271990530632, -92.4608802024137759, -74.7730327904656917, -361.4366744275644123, -163.2594078698224109, -131.7826859189169113, -87.9890592961395441, -71.4634683164663613, -375.6385028316967691, -188.0712743702100624, -125.4855200710422878, -91.3355397145749919, -68.0918516459060754, -441.1318905680280409, -180.4647430516490658, -119.2400966902216766, -82.2686239651064142, -64.7833925450403001, -285.9138689309418169, -155.5957559766566760, -112.8981334436850403, -78.1367985978475019, -61.4362018774827519, -297.4158837500489767, -161.8779039221794562, -107.5302652730988058, -62.5751807372716939, -58.1544671072481592, -360.5698302818472598, -139.3563439524956209, -101.3550944250247596, -66.1417784932326356, -54.8261503628728661, -329.4162354957296088, -133.5217515589868356, -95.1832888594590685, -52.8677413634549822, -51.4657653840344267, -248.3899165391785573, -139.3814407764575094, -88.5212741537597196, -55.7005368848646469, -48.1543182483757590, -276.9315386026431156, -131.9893545028087090, -82.6006623537969915, -69.0420185657362850, -44.8367507457907664, -181.4408060486774161, -108.3444969662783137, -75.9079601363687573, -63.1898802064647356, -41.5016478301897394, -219.0420401833663959, -84.4988900904494074, -70.1062756651865584, -33.2783970388448864, -38.1772512756981541, -273.1900086807029311, -108.5094322092337791, -64.3362014327106806, -59.0886867765107837, -34.8772016445090287, -169.4933133973034387, -71.7928430719725839, -58.3828297919128545, -47.2403703301395694, -31.5344304985637365, -117.5037569248842146, -92.0323886716669222, -52.8384828364647490, -31.5615938934498104, -28.2586823448365294, -153.6837047427770813, -49.9009684773715776, -46.2772438147131595, -17.9501825001563589, -24.9060574152862984, -110.1098213427187602, -67.5478893877768769, -40.0556986478239310, -18.8519922474688144, -21.5648208998895683, -173.3275868071849573, -61.1077300943754835, -33.3655902391294035, -12.3287393955675952, -18.3059646560534190, -24.8691406901685923, -24.3306319752870763, -27.3268126581254194, -23.7390389076165604, -14.9587256098364918, -85.4809613248568212, -13.5477484224198861, -21.8129405614249663, -20.0181616010522418, -11.6020015901856652, 28.3085794491403249, 0.4710971628685456, -14.8693489385043449, -23.2295641672642716, -8.3390948286546802, -15.1198466367483206, 4.0314245196032061, -8.6116775765411262, -1.4947965761929254, -4.9560803825906987, -65.8025302291763410, -15.0944089570277242, -3.4246042101295018, -9.2910673027247412, -1.6790651133890293, -52.7174211257638490, 0.0346524814304416, 2.1866888450084616, 3.9031011758404759, 1.6876983441107427, 22.1394368332931890, 19.2938732984953276, 9.3054528117352504, 20.7370055373316120, 4.9950898893310409, 106.0901990846174385, 15.2149792487102786, 15.9362665104415555, 1.2510364228377284, 8.2943845989940570, 109.9248283142999298, 23.5591141499866694, 22.0269987438173303, 26.9714368782933036, 11.5995588367357954, 41.8300958146797939, 19.7966426040319448, 28.1073666954062276, 14.6854725966631463, 14.9378289022346706, 67.4472048614366031, 31.3207631901612444, 33.5498786040999164, 15.6387331105817680, 18.2720078036694105, 74.6727506718792711, 56.6346800410327091, 40.5141883807065710, 24.2646824070052958, 21.6217271812209759, 68.5259532664758808, 66.7086254590863064, 46.0474341769780153, 47.2692069564994313, 24.9419268881516558, 165.1707871115859234, 84.3109801754752652, 52.8153295478530254, 44.3882524017164712, 28.2661884361734650, 257.4051502537284932, 80.6920056720862249, 58.4039790397064849, 30.9839014819423824, 31.5521043756045998, 263.7439906029040912, 93.2881379011682839, 63.8828521356234091, 33.4438374216607315, 34.8599274359979034, 155.4379718588401147, 86.8130322691444007, 70.1873402014866770, 35.1788050705325475, 38.2248099698418144, 222.0850104428774330, 109.7558062739266802, 76.7447221692349331, 42.3067891409962442, 41.4939268423117795, 198.4902524012832146, 136.4035021496149511, 81.9660149260858475, 55.7815783782407095, 44.8718440796001659, 290.6826916506388443, 102.7612970413367748, 89.4292043670429280, 58.8306294641704426, 48.1443809733186328, 267.1191911986312562, 136.2637212973556871, 94.1064016030773018, 71.3320432463911231, 51.4977058375590531, 267.3487586489829937, 137.4064501257937536, 100.4988806851764593, 55.8532264308257425, 54.8181056284801542, 323.3171998564300793, 163.9317509418108330, 106.6353964439066999, 73.4388162582944233, 58.1462153329587608, 277.4672444686221979, 151.2315318037415750, 113.5921992018054425, 71.5017957816106389, 61.4471303980954389, 383.6026992996435183, 162.0302841272167598, 120.0442700950936228, 95.2966849241175851, 64.8023963082147674, 376.5067514457421112, 160.9494128390791730, 125.7659754028471184, 85.8592558438856486, 68.1015762672011675, 433.8847101181395374, 201.9773739841361930, 132.1816551284973684, 80.4479269137954702, 71.4520397349944432, 457.1683498114880422, 183.4636090726505699, 138.1753951578972988, 95.8893853579812969, 74.7499366085014145, 457.9370858196487006, 184.6147858378209889, 143.6754463205904813, 98.9155439303665673, 78.0368099551707672, 502.8151295897334876, 200.6804842178709123, 149.9982408100326552, 119.4560214261871920, 81.4159841853305721, 444.3380955508980605, 212.3207886860920439, 156.2861250351164699, 97.4167936930100495, 84.7331799259197709, 471.8482637947902276, 208.4009583839196580, 162.6448614307224716, 114.2452622523234140, 88.0013776272131594, 531.6707574035544894, 245.0303450652081096, 167.6198867670972845, 123.0478537746750476, 91.3392902313499775, 528.6204101631115009, 223.4087640715310386, 174.6008926377062380, 136.2504226985812466, 94.6653719341121729, 483.6767369781578054, 265.5771921363370325, 181.4356752970279558, 126.3039083261327278, 98.0300322996041729, 647.4593380083251759, 269.0846489720924524, 187.3646539910617435, 120.1263991398942039, 101.3478473539227451, 581.7159444786464064, 258.0741358305621702, 193.0326094656455780, 146.4832370625846352, 104.6411106839718315, 542.5812369035263600, 266.5479394834283653, 198.6884912030702708, 145.6603378209440507, 108.0000233546161184, 642.4300057251925864, 273.5332520662976208, 205.5274548759527988, 132.4886275800243993, 111.2594829999989798, 678.2087501698360938, 287.2884271595189034, 211.2246187844340852, 134.4187639643407124, 114.6110991188589736, 689.6774466467077218, 322.1006769953833668, 216.6201798342207780, 152.7504315132451893, 117.9338364401240966
            });
        }

        private static NDArray betaGradient() {
            return ProviderStore.array(new double[]{
                    12780.0000000000000000, 12852.0000000000000000, 12924.0000000000000000, 12996.0000000000000000, 13068.0000000000000000
            });
        }

        private static NDArray gammaGradient() {
            return ProviderStore.array(new double[]{
                    -871.6142249750200790, -547.9235128663648311, -27.2313074338042362, -783.0401210378884116, -2.2919204949031147
            });
        }

        private static NDArray testOutput() {
            return ProviderStore.array(new double[]{
                    1.8308729572186548, 0.2550476051258954, 1.5744377348218244, 0.7951995974060146, 0.4323652703339462, -0.9958353939579183, 1.7048865365299783, -1.9784845789650731, -0.7831132502530860, -0.4411754855727956, 1.1245277560099893, -0.9261581024862000, 1.1711862596989597, 1.7130856670477623, -1.3413340233687272, 0.4392544576148110, -1.5638182662228175, -0.5802576827463835, -1.4967618536971870, 1.2925733823693761, -0.1674907820940188, 1.6126427727175345, 0.8892651222411985, 1.3128584403628878, -1.3103601827375826, -1.5271215186528373, 0.4358726261229666, -1.3838815822359276, 0.8586425172946317, -0.5287087728590016, 0.1779226891766044, 1.5741000354651637, -0.0835056320154344, -1.2779284630377319, -1.3600011384471244, 0.2430205485010211, 0.1779056179919243, 0.3426231074405433, -0.2755709487992473, 0.2731511916360390, -0.3152239772253891, 0.5141812067307709, 0.8197803414567666, -1.0360365612747078, -0.6379044571645416, 0.0824026057229341, 0.7262321258155078, -1.0687999211884556, 0.2481440815084071, 0.1905129168199480, 1.5616229442639460, -0.3680131496793377, 0.3424276374099531, -0.9934119164783901, -0.2318657949187291, -1.1224793196105987, 1.5092572322605231, -0.4139046856529234, 1.1894828817958931, -0.5010263934239845, 1.5961274209337128, -0.2261807127602813, -0.7839098305458396, 0.3076012946716560, 0.9231093499336396, 1.4318066680220660, 0.4187918773877595, -1.6901516795484941, 0.3934754825981646, -1.0702011531123468, 0.9484016551700787, 1.4290618342201689, -0.1091002085271455, 0.4149950998299019, -1.6694863697797615, 0.2237178965139650, -1.1576306420832907, 0.2578952207995329, -0.4049062722827212, 0.6650191031013044, -1.6265514517193824, -1.2230821174881186, 0.5132485716963338, 0.1111465035613595, 0.0134725067014116, 1.3666580312717560, 0.0540007134716438, 0.9769010601009640, 0.0960751311484778, 1.1931311866581744, 0.7012212774915794, -1.0915999970835779, -0.6612016452291467, 1.3111005416943424, -0.7219257564419976, -1.0977093837965954, 0.0029189237909879, -0.5574285961104710, 0.4675097081219501, -0.4346148607081730, -0.8270751716833100, -0.2003396417075569, -0.4609166451791094, 1.4363411742581871, 1.3688629543679767, 0.5380085048267400, -1.3130849263931550, 0.6932945301189068, 0.6717235826389860, 0.8585888534027752, -0.5013598785629965, -1.3952142044053693, 0.2478185226626266, -1.2238595138601660, 0.6376820351487886, 1.1811431885623511, -0.2133342764397046, 1.4682425746616570, -1.0537874133364884, 1.2458390421107701, -0.0570370252200831, 0.9841583128292548, 0.7661631440767831, 1.7055987684488549, 1.3478057079940449, -1.6583328470216059, -1.5402136287827304, -0.0041200840802169, -1.5319020126417566, 0.2986644218204972, 0.2042488435091370, 0.6582799470431420, -0.3789118561666776, -0.7165094148910551, 1.2693707668580956, 0.9321251150261745, -1.5728143597124182, -1.6362353010066979, 0.5111248323994886, -0.9287298192496642, -0.2748651833456606, 1.0468010170790194, -0.6994626718718724, 1.5162651813475936, 0.5078575130725511, 0.2683304026907134, -0.9826598930144472, -0.4956301236015150, 0.9594640824639313, 1.4060074448833824, -1.5320007555191451, -1.1388223587821216, 0.7191973997709127, 1.2017628466705599, -1.5907325240933226, 1.3128708548062020, 1.0643771166615341, 0.5286816627818887, -0.4859826207058351, -0.4087865300955222, -0.4302732343065037, 1.2459566491169434, -0.7943958282827701, -0.5452810329113564, 1.2216033271850542, 1.6537944392973785, 1.6792031578077653, 0.9673594798384908, -1.3506450604863485, -1.5836305910863473, 0.2877373706607127, 1.2990706366730302, 1.2491396740843221, 0.5287468841540939, 1.2897548813363144, -1.2375132927652865, -0.8454081413464358, -0.7790547461674624, -0.7700496862773818, -0.8484372440307588, -1.3633892935354739, -0.3258180155375854, -1.8919320109606783, 0.1901861582871494, 1.2566112220006787, -0.1336935148520908, 0.5149825298529169, 0.2477837165109200, 1.5421375482585198, 0.5545912340549428, 1.2955666318801491, -0.4592668977565264, 1.3346740726030468, -1.0147273894856532, -0.5302422897410146, 0.9666902353702209, -0.4673535272631089, 1.2562582354667722, 1.2936062150668395, -1.3370952472993132, -0.9406652341899564, -1.4169941696167383, 1.1554799168718528, -0.4883800197287318, -0.5791972900136884, -0.7915283726400073, -1.1777672108424264, -0.3215678071526078, -0.8455348029127023, -0.0147271194302157, -1.0459916125125606, 0.1339186624925937, 1.4848900030028691, -0.3769397053773567, 1.2844874545670564, -1.5939089219943439, 0.2603627159346589, 0.2036141539960674, 1.6391059843427627, 1.1880296813168556, 0.1139207251597165, 0.9723056055781649, 1.5862786327172316, 0.8693054226864838, 1.2836133221660617, 1.7249632531240477, 0.0338250035585750, 0.4245445079387640, -1.0330437851455034, -0.4337604454766968, 1.4510413946718286, 0.3564243647183440, -0.9740485395519300, -1.2280471910999253, -1.1153786544831661, -1.3387447306219489, -0.8041830353519552, -0.5912543344513779, -1.5010731125846046, 0.9007408676079613, -0.2892122088538809, 0.3230945092329884, 0.3371961262583536, -1.1936919834143591, -1.6108827247826554, -1.2200224019855970, 1.7385109365049156, -1.6171648629886679, -0.2032551800236213, 1.0215097695126651, 0.3901001124288577, -1.5349339164696305, 1.2657001187914219, -0.3348568534819716, -1.3284172665431200, -0.5400241375796302, 0.4135878893911054, -1.8626279861895814, 0.5508233660481370, 0.1412625587311034, -0.9480130750452602, -0.1545659080354218, -1.2899800594742779, -1.5747839478485499, 0.0542646367144854, -0.1328207737828633, 1.2513316206014271, -1.2696111223344442, -0.1419346873308438, 0.3317868101311738, -1.5520179403706598, -0.3934148012164478, 0.5206491769738228, -0.8101472041058315, -0.6764372773203391, 0.3640819632732422, -0.2106011240227863, 1.2218753110560403, 1.2909590800037343, 0.8850140345900392, -0.2046643106829986, -0.9516885690287955, 0.3472289244552798, -0.1844621852673729, -0.2052448324698126, 0.6414596402984065, 1.5821086802509181, 0.9699356947939854, -1.2265866216192460, 1.1291498825101129, 0.7393888662819412, -0.5147683510559560, 0.6822447329897194, -0.0244930041597371, -0.0217724975617060, 0.3432319020868491, -1.0822651362334019, -0.6706534814818941, -0.1585584150202610, -1.6938806243859403, 0.9150470921694227, -0.4898309583129725, -0.2483604412208749, 1.5923016770125002, 0.9979417400431529, -0.7812494247423345, -0.2415681561015628, 0.0986085046664780, -1.2393499827776289, 0.7594582372800467, -0.5905696625620562, -1.2034468908569815, 0.5984515601339258, 0.1120163954557309, -1.7956232178362761, 0.3091992415737650, 0.9882647826489350, -1.8872680642322130, 0.5996220773018450, -1.0546258196596412, -0.1707669326744368, -1.3503130684650611, -0.0447859447964483, 1.5607619943414814, -0.8729863527987405, -1.5700758720196966, 1.2721804814245283, 1.4821978199557613, 0.0305519671069909, 1.1326291826746093, 1.6110820233643333, 0.8879361184283807, 1.0547746324997753, -1.0940302976617726, 0.9234264621208261, -0.2446736645702563, -0.6253986259299671, 0.0641549761271776, 1.2827979607925961, -0.4465670775502311, -1.5165057233935002, -0.6234006678246233, -0.9525155662510247, 0.7344893659701732, 1.2873008878891881, 0.2616337970907363, -0.7371655162622464, 0.5834893579174771, -1.1428225937392860, -1.6809106173428141, 0.6337662259292589, -0.3244251676549439, -0.3441087524475641, -1.3948440714648314, -0.2920192502657919, 0.4724182920976983, 1.7259642867668257, -1.9224599467331736, 0.1182995430852856, -0.2685001375932785,
            });
        }

        private static NDArray testInput() {
            return ProviderStore.array(new double[]{
                    1.9670298, -2.9055355, 11.918053, -13.140736, 28.488644, 1.2160895,
                    -2.047451, 9.018691, -14.9880705, 27.173958, 1.779383, -3.6046298,
                    11.58898, -12.066398, 25.819212, 1.5973339, -3.9820278, 10.159714,
                    -15.8233595, 29.783264, 1.4361466, -2.1020453, 11.358918, -12.5348425,
                    25.865828, 1.0749485, -2.7985146, 9.503917, -13.06648, 27.04222,
                    1.5279088, -2.124857, 10.565088, -15.567226, 25.791117, 1.5452026,
                    -2.951192, 10.91283, -14.394018, 28.249025, 1.3969, -2.7521677,
                    11.302215, -15.284104, 26.877878, 1.5025331, -2.6266658, 9.761039,
                    -13.781036, 28.124655, 1.8955011, -3.2742927, 10.912671, -15.234215,
                    27.48897, 1.1824454, -2.163234, 10.295466, -12.679248, 27.083881,
                    1.9046676, -3.1903496, 9.993524, -13.711445, 29.227219, 1.8610144,
                    -2.8086238, 9.253984, -13.610933, 26.227268, 1.7325934, -2.2106974,
                    10.544202, -13.585746, 25.325338, 1.5400747, -3.7416265, 10.843689,
                    -14.545399, 28.83879, 1.0485342, -3.7803638, 11.05207, -13.941385,
                    27.858208, 1.843707, -3.0245247, 11.430433, -13.959025, 29.633604,
                    1.6669278, -3.7025464, 10.093659, -12.5369005, 26.751427, 1.1890258,
                    -3.0547574, 10.178344, -13.524281, 27.183832, 1.2609222, -3.1750555,
                    10.257102, -12.390312, 29.89808, 1.6235689, -3.8336318, 11.198997,
                    -13.285258, 29.130114, 1.3474513, -3.8822398, 10.835465, -15.503942,
                    28.797647, 1.7944233, -3.1827464, 11.831392, -15.304881, 29.712929,
                    1.4654896, -2.4740126, 11.258461, -12.07516, 29.86639, 1.0400912,
                    -3.9680574, 10.629871, -15.86449, 28.287424, 1.5349026, -2.6668832,
                    10.324022, -14.910114, 29.748344, 1.7282695, -3.9873521, 9.297983,
                    -13.473231, 26.440184, 1.4076216, -2.4369378, 10.062437, -12.296765,
                    28.60226, 1.5519265, -3.6380703, 10.228774, -12.948473, 29.953983,
                    1.0736524, -3.7304947, 11.220134, -12.664874, 25.443865, 1.829418,
                    -2.4265354, 11.064664, -14.640294, 27.222704, 1.3663361, -2.319068,
                    9.984966, -14.7097, 29.676455, 1.9199873, -2.0626516, 11.4226465,
                    -15.652337, 25.454554, 1.557082, -2.2876325, 11.652594, -13.452605,
                    29.779022, 1.1518856, -3.556838, 9.997485, -14.97278, 26.561026,
                    1.1184455, -3.2493198, 9.089322, -13.848873, 29.729141, 1.4451252,
                    -2.7516935, 10.835437, -12.266483, 28.672596, 1.8248209, -3.3283012,
                    11.722394, -15.259163, 27.039911, 1.737452, -3.3330872, 11.658402,
                    -12.557377, 25.82559, 1.2307459, -3.8951302, 11.576162, -14.643101,
                    26.966234, 1.2703655, -3.7535443, 10.370818, -15.0611315, 27.815765,
                    1.2027651, -2.9772255, 11.844977, -14.512665, 29.771095, 1.0572059,
                    -2.9023898, 10.799393, -12.152987, 29.625925, 1.5109061, -2.4810276,
                    11.927716, -13.053999, 29.76978, 1.938894, -3.0364656, 10.979683,
                    -15.2806015, 27.185118, 1.8661242, -2.8455358, 9.838361, -15.508843,
                    26.159277, 1.1249926, -3.5324392, 10.15074, -15.828406, 29.193554,
                    1.4038103, -2.865262, 10.908402, -15.468632, 25.413538, 1.1565322,
                    -2.0275505, 9.313546, -14.309377, 29.37531, 1.5842756, -3.9649327,
                    11.666108, -14.463409, 25.838652, 1.3371798, -2.8117037, 9.113235,
                    -13.426765, 28.050531, 1.228794, -3.1479645, 9.580544, -15.9146805,
                    27.9196, 1.4453571, -2.3158867, 9.597167, -14.237604, 28.337273,
                    1.0683346, -3.2893267, 11.058109, -15.019712, 26.819887, 1.5773637,
                    -3.1811287, 11.630344, -12.560475, 29.169884, 1.4262712, -3.61974
                    , 10.91659, -14.28738, 27.529036, 1.6510515, -2.120117, 11.424749,
                    -15.5071335, 29.537312, 1.6770673, -3.3611495, 11.189979, -14.100144,
                    27.805162, 1.5718247, -3.6970215, 10.085946, -14.257061, 25.288626,
                    1.7237325, -3.3463905, 10.430558, -12.207768, 29.339842, 1.2730962,
                    -3.1994567, 10.713702, -15.522073, 28.980923, 1.3237519, -3.7687428,
                    11.121599, -13.940367, 25.135502, 1.5627836, -2.4715824, 9.093128,
                    -13.36965, 26.25071, 1.4352763, -3.8556652, 10.596685, -12.244684,
                    26.524078, 1.0635374, -2.3035474, 11.842781, -14.035717, 29.542547,
                    1.9086404, -2.5309618, 11.493982, -15.351983, 29.227695, 1.4156424,
                    -3.426626, 10.685587, -12.570027, 27.165844, 1.0777688, -3.4254434,
                    9.855933, -13.211794, 29.77533, 1.5501474, -3.492775, 11.109389,
                    -15.409092, 25.308146, 1.6490077, -3.2484953, 10.352423, -15.70407,
                    27.39844, 1.6061442, -2.0349762, 9.064409, -13.933013, 27.433836
            });
        }
    }
}
