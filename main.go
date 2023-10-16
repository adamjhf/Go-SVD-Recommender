package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"time"

	"github.com/jackc/pgx/v5"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

type Ratings struct {
	Users   []int
	Items   []int
	Ratings []float64
	UserMap map[string]int
	ItemMap map[string]int
}

type Trainset struct {
	Ratings    *Ratings
	NumUsers   int
	NumItems   int
	NumRatings int
	GlobalMean float64
}

type Prediction struct {
	FilmID     string
	Prediction float64
}

func main() {
	ratings := &Ratings{
		UserMap: make(map[string]int),
		ItemMap: make(map[string]int),
	}
	start := time.Now()
	loadRatings(ratings, "host="+os.Getenv("PGHOST"), 0)
	loadRatings(ratings, "host="+os.Getenv("PGHOST")+" dbname=soothsayer", 0)
	log.Printf("loading ratings took %s\n", time.Since(start))
	trainset := createTrainset(ratings)
	config := SVDConfig{
		NumEpochs:  20,
		NumFactors: 20,
	}
	start = time.Now()
	m := svd(trainset, &config)
	log.Printf("svd took %s\n", time.Since(start))
	user := "freeth"
	var predictions []Prediction
	start = time.Now()
	for film := range trainset.Ratings.ItemMap {
		predictions = append(predictions, Prediction{film, m.predict(user, film)})
	}
	log.Printf("predictions took %s\n", time.Since(start))
	sort.Slice(predictions, func(i, j int) bool {
		return predictions[i].Prediction > predictions[j].Prediction
	})
	for _, pred := range predictions[:50] {
		fmt.Printf("%s: %f\n", pred.FilmID, pred.Prediction)
	}
}

func createTrainset(ratings *Ratings) *Trainset {
	//userCats := createCategoryCodes(ratings.Users)
	//filmCats := createCategoryCodes(ratings.Items)
	//userCodes := reverseMap(userCats)
	//filmCodes := reverseMap(filmCats)
	return &Trainset{
		Ratings: ratings,
		//Users:      userCats,
		//Items:      filmCats,
		//UserCodes:  userCodes,
		//ItemCodes:  filmCodes,
		NumUsers:   len(ratings.UserMap),
		NumItems:   len(ratings.ItemMap),
		NumRatings: len(ratings.Ratings),
		GlobalMean: stat.Mean(ratings.Ratings, nil),
	}
}

type SVDConfig struct {
	NumEpochs  int
	NumFactors int
	InitMean   float64
	InitStdDev float64
	LR         float64
	Reg        float64
}

type Model struct {
	PU         *mat.Dense
	QI         *mat.Dense
	BU         *[]float64
	BI         *[]float64
	GlobalMean float64
	Trainset   *Trainset
}

func svd(trainset *Trainset, config *SVDConfig) *Model {
	if config == nil {
		config = &SVDConfig{}
	}
	if config.NumEpochs == 0 {
		config.NumEpochs = 20
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

	numFactors := config.NumFactors
	lr := config.LR
	reg := config.Reg
	numRatings := trainset.NumRatings
	globalMean := trainset.GlobalMean

	bu := make([]float64, trainset.NumUsers)
	bi := make([]float64, trainset.NumItems)
	pu := randMat(config.InitMean, config.InitStdDev, trainset.NumUsers, config.NumFactors)
	qi := randMat(config.InitMean, config.InitStdDev, trainset.NumItems, config.NumFactors)

	for epoch := 0; epoch < config.NumEpochs; epoch++ {
		log.Printf("starting epoch %d\n", epoch)
		for idx := 0; idx < numRatings; idx++ {
			// 3.5s
			u := trainset.Ratings.Users[idx]
			i := trainset.Ratings.Items[idx]
			r := trainset.Ratings.Ratings[idx]
			//3s
			var dot float64
			for f := 0; f < numFactors; f++ {
				dot += pu.At(u, f) * qi.At(i, f)
			}
			// 6s
			err := r - (globalMean + bu[u] + bi[i] + dot)
			bu[u] += lr * (err - reg*bu[u])
			bi[i] += lr * (err - reg*bi[i])
			// 7s
			for f := 0; f < numFactors; f++ {
				puf := pu.At(u, f)
				qif := qi.At(i, f)
				pu.Set(u, f, lr*(err*qif-reg*puf))
				qi.Set(i, f, lr*(err*puf-reg*qif))
			}
		}
	}
	return &Model{
		PU:         pu,
		QI:         qi,
		BU:         &bu,
		BI:         &bi,
		GlobalMean: globalMean,
		Trainset:   trainset,
	}
}

func (m *Model) predict(u, i string) float64 {
	uid := m.Trainset.Ratings.UserMap[u]
	iid := m.Trainset.Ratings.ItemMap[i]
	return m.GlobalMean + (*m.BU)[uid] + (*m.BI)[iid] + mat.Dot(m.PU.RowView(uid), m.QI.RowView(iid))
}

func randMat(mean, stdDev float64, r, c int) *mat.Dense {
	data := make([]float64, r*c)
	for i := range data {
		data[i] = rand.NormFloat64()*stdDev + mean
	}
	return mat.NewDense(r, c, data)
}

func loadRatings(ratings *Ratings, connString string, limit int) {
	ctx := context.Background()
	conn, err := pgx.Connect(ctx, connString)
	if err != nil {
		log.Fatalf("Unable to connect to database: %v\n", err)
	}
	defer conn.Close(ctx)
	queryFilter := ""
	if limit > 0 {
		queryFilter += fmt.Sprintf(" LIMIT %d", limit)
	}
	rows, err := conn.Query(ctx, "SELECT user_name, film_id, rating FROM ratings"+queryFilter)
	if err != nil {
		log.Fatalf("Unable to get ratings: %v\n", err)
	}
	j := 0
	var u, i string
	var r float64
	for rows.Next() {
		if j%1000000 == 0 {
			log.Printf("loaded %d rows", j)
		}
		rows.Scan(&u, &i, &r)
		uid, ok := ratings.UserMap[u]
		if !ok {
			uid = len(ratings.UserMap)
			ratings.UserMap[u] = uid
		}
		iid, ok := ratings.ItemMap[i]
		if !ok {
			iid = len(ratings.ItemMap)
			ratings.ItemMap[i] = iid
		}
		ratings.Users = append(ratings.Users, uid)
		ratings.Items = append(ratings.Items, iid)
		ratings.Ratings = append(ratings.Ratings, r)
		j++
	}
}

func loadRatingsFromCSV(fileName string) Ratings {
	f, _ := os.Open(fileName)
	defer f.Close()
	r := csv.NewReader(f)
	var ratings Ratings
	for {
		record, err := r.Read()
		if err != nil {
			break
		}
		u := record[0]
		i := record[1]
		uid, ok := ratings.UserMap[u]
		if !ok {
			uid = len(ratings.UserMap)
			ratings.UserMap[u] = uid
		}
		iid, ok := ratings.ItemMap[i]
		if !ok {
			iid = len(ratings.ItemMap)
			ratings.ItemMap[i] = iid
		}
		ratings.Users = append(ratings.Users, uid)
		ratings.Items = append(ratings.Items, iid)
		rating, _ := strconv.ParseFloat(record[2], 32)
		ratings.Ratings = append(ratings.Ratings, rating)
	}
	return ratings
}

func createCategoryCodes(strings []string) map[string]int {
	codes := make(map[string]int)
	code := 0
	for _, str := range strings {
		if _, ok := codes[str]; !ok {
			codes[str] = code
			code++
		}
	}
	return codes
}

func reverseMap(m map[string]int) map[int]string {
	r := make(map[int]string, len(m))
	for k, v := range m {
		r[v] = k
	}
	return r
}
