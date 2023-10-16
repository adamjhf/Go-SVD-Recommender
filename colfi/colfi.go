package colfi

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"log"
	"math/rand"
)

type Dataset struct {
	Users   []int
	Items   []int
	Ratings []float64
	UserMap map[string]int
	ItemMap map[string]int
}

type Model interface {
	Fit(numEpochs int)
	Predict(u, i string) float64
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

func (d *Dataset) Append(u, i string, r float64) {
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
	d.Users = append(d.Users, uid)
	d.Items = append(d.Items, iid)
	d.Ratings = append(d.Ratings, r)
}

func NewSVD(dataset *Dataset, config *SVDConfig) *SVD {
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
		GlobalMean: stat.Mean(dataset.Ratings, nil),
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
			r := m.Dataset.Ratings[idx]
			var dot float64
			for f := 0; f < numFactors; f++ {
				dot += pu.At(u, f) * qi.At(i, f)
			}
			err := r - (globalMean + bu[u] + bi[i] + dot)
			bu[u] += lr * (err - reg*bu[u])
			bi[i] += lr * (err - reg*bi[i])
			for f := 0; f < numFactors; f++ {
				puf := pu.At(u, f)
				qif := qi.At(i, f)
				pu.Set(u, f, lr*(err*qif-reg*puf))
				qi.Set(i, f, lr*(err*puf-reg*qif))
			}
		}
	}
}

func (m *SVD) Predict(u, i string) float64 {
	uid := m.Dataset.UserMap[u]
	iid := m.Dataset.ItemMap[i]
	return m.GlobalMean + (*m.BU)[uid] + (*m.BI)[iid] + mat.Dot(m.PU.RowView(uid), m.QI.RowView(iid))
}

func randMat(mean, stdDev float64, r, c int) *mat.Dense {
	data := make([]float64, r*c)
	for i := range data {
		data[i] = rand.NormFloat64()*stdDev + mean
	}
	return mat.NewDense(r, c, data)
}
