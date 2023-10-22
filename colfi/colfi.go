package colfi

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

type Dataset struct {
	Users   []int
	Items   []int
	Ratings []float32
	UserMap map[string]int
	ItemMap map[string]int
}

type Model interface {
	Fit(numEpochs int)
	Predict(u, i string) float64
	GetDataset() *Dataset
}

type SVD struct {
	Dataset    *Dataset
	PU         *mat.Dense
	QI         *mat.Dense
	BU         *[]float64
	BI         *[]float64
	GlobalMean float64
	Config     *SVDConfig
}

type SVDConfig struct {
	NumFactors int
	InitMean   float64
	InitStdDev float64
	LR         float64
	Reg        float64
	Verbose    bool
}

func NewDataset() *Dataset {
	return &Dataset{
		UserMap: make(map[string]int),
		ItemMap: make(map[string]int),
	}
}

func DatasetsFromSlices(u, i []string, r []float32, split float64) (*Dataset, *Dataset, error) {
	n := len(u)
	if n != len(i) || len(u) != len(r) {
		return nil, nil, fmt.Errorf("u, i and r slices must be the same length")
	}
	if split < 0.0 || split > 1.0 {
		return nil, nil, fmt.Errorf("split must be between 0 and 1")
	}
	p := rand.Perm(n)
	trainNum := int(math.Round(float64(n) * (1. - split)))
	trainset := NewDataset()
	for _, j := range p[:trainNum] {
		trainset.Append(u[p[j]], i[p[j]], r[p[j]])
	}
	testset := NewDataset()
	for _, j := range p[trainNum:] {
		testset.Append(u[p[j]], i[p[j]], r[p[j]])
	}
	return trainset, testset, nil
}

func (d *Dataset) Append(u, i string, r float32) {
	uid, iid := d.getInternalIDs(u, i)
	d.Users = append(d.Users, uid)
	d.Items = append(d.Items, iid)
	d.Ratings = append(d.Ratings, r)
}

func NewSVD(dataset *Dataset, config *SVDConfig) Model {
	if config == nil {
		config = &SVDConfig{}
	}
	if config.NumFactors == 0 {
		config.NumFactors = 50
	}
	if config.InitStdDev == 0 {
		config.InitStdDev = .1
	}
	if config.LR == 0 {
		config.LR = .005
	}
	if config.Reg == 0 {
		config.Reg = .02
	}
	bu := make([]float64, len(dataset.UserMap))
	bi := make([]float64, len(dataset.ItemMap))
	svd := &SVD{
		Dataset:    dataset,
		PU:         randMat(config.InitMean, config.InitStdDev, len(dataset.UserMap), config.NumFactors),
		QI:         randMat(config.InitMean, config.InitStdDev, len(dataset.ItemMap), config.NumFactors),
		BU:         &bu,
		BI:         &bi,
		GlobalMean: mean32(dataset.Ratings),
		Config:     config,
	}
	return svd
}

func (m *SVD) Fit(numEpochs int) {
	numRatings := len(m.Dataset.Ratings)
	numFactors := m.Config.NumFactors
	reg := m.Config.Reg
	lr := m.Config.LR
	pu := m.PU
	qi := m.QI
	bu := *m.BU
	bi := *m.BI
	globalMean := m.GlobalMean
	for epoch := 0; epoch < numEpochs; epoch++ {
		if m.Config.Verbose {
			log.Printf("running epoch %d\n", epoch)
		}
		for idx := 0; idx < numRatings; idx++ {
			u := m.Dataset.Users[idx]
			i := m.Dataset.Items[idx]
			r := float64(m.Dataset.Ratings[idx])
			dot := float64(0)
			for f := 0; f < numFactors; f++ {
				dot += pu.At(u, f) * qi.At(i, f)
			}
			err := r - (globalMean + bu[u] + bi[i] + dot)
			bu[u] += lr * (err - reg*bu[u])
			bi[i] += lr * (err - reg*bi[i])
			for f := 0; f < numFactors; f++ {
				puf := pu.At(u, f)
				qif := qi.At(i, f)
				pu.Set(u, f, puf+lr*(err*qif-reg*puf))
				qi.Set(i, f, qif+lr*(err*puf-reg*qif))
			}
		}
	}
}

func (m *SVD) Predict(u, i string) float64 {
	p := m.GlobalMean
	uid, uok := m.Dataset.UserMap[u]
	if uok {
		p += (*m.BU)[uid]
	}
	iid, iok := m.Dataset.ItemMap[i]
	if iok {
		p += (*m.BI)[iid]
	}
	if uok && iok {
		p += mat.Dot(m.PU.RowView(uid), m.QI.RowView(iid))
	}
	return p
}

func (m *SVD) GetDataset() *Dataset {
	return m.Dataset
}

type SVDpp struct {
	Dataset    *Dataset
	PU         *mat.Dense
	QI         *mat.Dense
	YJ         *mat.Dense
	BU         *[]float64
	BI         *[]float64
	IU         map[int][]int
	GlobalMean float64
	Config     *SVDConfig
}

func NewSVDpp(dataset *Dataset, config *SVDConfig) Model {
	if config == nil {
		config = &SVDConfig{}
	}
	if config.NumFactors == 0 {
		config.NumFactors = 50
	}
	if config.InitStdDev == 0 {
		config.InitStdDev = .1
	}
	if config.LR == 0 {
		config.LR = .005
	}
	if config.Reg == 0 {
		config.Reg = .02
	}
	bu := make([]float64, len(dataset.UserMap))
	bi := make([]float64, len(dataset.ItemMap))

	if config.Verbose {
		log.Println("caching user ratings")
	}
	iu := make(map[int][]int, len(dataset.UserMap))
	avgNum := len(dataset.Ratings) / len(dataset.Users)
	for idx := range dataset.Ratings {
		uid := dataset.Users[idx]
		if _, ok := iu[uid]; !ok {
			iu[uid] = make([]int, 0, avgNum)
		}
		iu[uid] = append(iu[uid], dataset.Items[idx])
	}

	svd := &SVDpp{
		Dataset:    dataset,
		PU:         randMat(config.InitMean, config.InitStdDev, len(dataset.UserMap), config.NumFactors),
		QI:         randMat(config.InitMean, config.InitStdDev, len(dataset.ItemMap), config.NumFactors),
		YJ:         randMat(config.InitMean, config.InitStdDev, len(dataset.ItemMap), config.NumFactors),
		BU:         &bu,
		BI:         &bi,
		IU:         iu,
		GlobalMean: mean32(dataset.Ratings),
		Config:     config,
	}
	return svd
}

func (m *SVDpp) Fit(numEpochs int) {
	numRatings := len(m.Dataset.Ratings)
	numFactors := m.Config.NumFactors
	reg := m.Config.Reg
	lr := m.Config.LR
	pu := m.PU
	qi := m.QI
	yj := m.YJ
	bu := *m.BU
	bi := *m.BI
	iu := m.IU
	globalMean := m.GlobalMean

	for epoch := 0; epoch < numEpochs; epoch++ {
		if m.Config.Verbose {
			log.Printf("running epoch %d", epoch)
		}
		for idx := 0; idx < numRatings; idx++ {
			u := m.Dataset.Users[idx]
			i := m.Dataset.Items[idx]
			r := float64(m.Dataset.Ratings[idx])

			uImpFdb := make([]float64, numFactors)
			sqrtU := math.Sqrt(float64(len(iu[u])))
			for _, item := range iu[u] {
				for f := 0; f < numFactors; f++ {
					uImpFdb[f] += yj.At(item, f) / sqrtU
				}
			}

			dot := float64(0)
			for f := 0; f < numFactors; f++ {
				dot += (pu.At(u, f) + uImpFdb[f]) * qi.At(i, f)
			}
			err := r - (globalMean + bu[u] + bi[i] + dot)
			bu[u] += lr * (err - reg*bu[u])
			bi[i] += lr * (err - reg*bi[i])

			for f := 0; f < numFactors; f++ {
				puf := pu.At(u, f)
				qif := qi.At(i, f)
				pu.Set(u, f, puf+lr*(err*qif-reg*puf))
				qi.Set(i, f, qif+lr*(err*(puf+uImpFdb[f])-reg*qif))
				errQIF := err * qif / sqrtU
				for _, item := range iu[u] {
					yj.Set(item, f, yj.At(item, f)+lr*(errQIF-reg*yj.At(item, f)))
				}
			}
		}
	}
}

func (m *SVDpp) Predict(u, i string) float64 {
	p := m.GlobalMean
	uid, uok := m.Dataset.UserMap[u]
	if uok {
		p += (*m.BU)[uid]
	}
	iid, iok := m.Dataset.ItemMap[i]
	if iok {
		p += (*m.BI)[iid]
	}
	if uok && iok {
		uImp := mat.NewVecDense(m.Config.NumFactors, nil)
		for _, item := range m.IU[uid] {
			uImp.AddVec(uImp, m.YJ.RowView(item))
		}
		uImp.ScaleVec(1.0/math.Sqrt(float64(len(m.IU[uid]))), uImp)
		uImp.AddVec(uImp, m.QI.RowView(iid))
		p += mat.Dot(m.PU.RowView(uid), uImp)
	}
	return p
}

func (m *SVDpp) GetDataset() *Dataset {
	return m.Dataset
}

type GridSearchParams struct {
	NumEpochs  []int
	NumFactors []int
	Reg        []float64
	LR         []float64
	InitStdDev []float64
}

type GridSearchTestResult struct {
	NumEpochs  int
	NumFactors int
	Reg        float64
	LR         float64
	InitStdDev float64
	Loss       float64
	Runtime    time.Duration
}

func GridSearch(
	trainset *Dataset,
	testset *Dataset,
	p GridSearchParams) []GridSearchTestResult {
	numTests := (len(p.NumEpochs) * len(p.NumFactors) * len(p.Reg) * len(p.LR) * len(p.InitStdDev))
	if numTests < 1 {
		log.Fatalln("GridSearch: all parameters must have at least one test value")
	}
	tests := make([]GridSearchTestResult, 0, numTests)
	userReverseMap := reverseMap(testset.UserMap)
	itemReverseMap := reverseMap(testset.ItemMap)
	i := 0
	for _, numEpochs := range p.NumEpochs {
		for _, numFactors := range p.NumFactors {
			for _, reg := range p.Reg {
				for _, lr := range p.LR {
					for _, initStdDev := range p.InitStdDev {
						i++
						log.Printf("running grid search test %d / %d", i, numTests)
						config := &SVDConfig{
							NumFactors: numFactors,
							Reg:        reg,
							LR:         lr,
							InitStdDev: initStdDev,
						}
						start := time.Now()
						loss := testModel(trainset, testset, numEpochs, config,
							userReverseMap, itemReverseMap)
						runtime := time.Since(start)
						test := GridSearchTestResult{
							NumEpochs:  numEpochs,
							NumFactors: numFactors,
							Reg:        reg,
							LR:         lr,
							InitStdDev: initStdDev,
							Loss:       loss,
							Runtime:    runtime,
						}
						tests = append(tests, test)
					}
				}
			}
		}
	}
	return tests
}

func testModel(trainset, testset *Dataset, numEpochs int, config *SVDConfig,
	userReverseMap, itemReverseMap map[int]string) float64 {
	m := NewSVD(trainset, config)
	m.Fit(numEpochs)
	actual := make([]float64, 0, len(testset.Ratings))
	pred := make([]float64, 0, len(testset.Ratings))
	for idx, r := range testset.Ratings {
		u, ok := userReverseMap[testset.Users[idx]]
		if !ok {
			log.Fatalf("user id %d not found in reverse map", testset.Users[idx])
		}
		i, ok := itemReverseMap[testset.Items[idx]]
		if !ok {
			log.Fatalf("item id %d not found in reverse map", testset.Items[idx])
		}
		actual = append(actual, float64(r))
		pred = append(pred, m.Predict(u, i))
	}
	return RMSE(pred, actual)
}

func RMSE(pred, actual []float64) float64 {
	n := len(pred)
	if n != len(actual) {
		log.Fatalf("pred and actual slices must be the same length")
	}
	var s float64
	for i := range pred {
		s += math.Pow(pred[i]-actual[i], 2.)
	}
	return math.Sqrt(s / float64(n))
}

func (d *Dataset) getInternalIDs(u, i string) (int, int) {
	uid, ok := d.UserMap[u]
	if !ok {
		uid = len(d.UserMap)
		d.UserMap[u] = uid
	}
	iid, ok := d.ItemMap[i]
	if !ok {
		iid = len(d.ItemMap)
		d.ItemMap[i] = iid
	}
	return uid, iid
}

func randMat(mean, stdDev float64, r, c int) *mat.Dense {
	data := make([]float64, r*c)
	for i := range data {
		data[i] = rand.NormFloat64()*stdDev + mean
	}
	return mat.NewDense(r, c, data)
}

func mean32(s []float32) float64 {
	var sum float64
	for _, x := range s {
		sum += float64(x)
	}
	return sum / float64(len(s))
}

func reverseMap(m map[string]int) map[int]string {
	r := make(map[int]string, len(m))
	for k, v := range m {
		r[v] = k
	}
	return r
}
