import {useState} from "react"
import './MainPage.css'
import {Slider} from "@mui/material"

const staticImageNames = Array.from({ length: 10 }, (_, i) => `${i + 1}.jpg`);
const staticImages = staticImageNames.map(
    (name) => `/raw_images/${name}`
);

function SLICShowcase() {
  const [file, setFile] = useState("/placeholders/Untitled.jpg");
  const [resultUrl, setResultUrl] = useState("/placeholders/UntitledSLIC.jpg");

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://localhost:8000/slic-processing", {
      method: "POST",
      body: formData,
    });

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);

    setResultUrl(url);
  };

  return (
    <div className="slic-showcase">
        <h2 className="slic-title">Try Our SLIC!</h2>

        <div className="slic-buttons">
            <label htmlFor="slic-browse-button" className="special-button"> Upload Photo</label>
            <input
                className="slic-browse-button"
                id="slic-browse-button"
                type="file"
                accept="image/*"
                onChange={(e) => setFile(e.target.files[0])}
            />

            <button onClick={handleUpload} className="slic-send-button special-button">
                Process
            </button>
        </div>
        
      <div className="slic-result">
        {file && (
          <div>
            <h4>Original:</h4>
            <img src={file == "/placeholders/Untitled.jpg"? "/placeholders/Untitled.jpg" : URL.createObjectURL(file)} className="slic-sended-image"/>
          </div>
        )}

        {resultUrl && (
          <div>
            <h4>Processed:</h4>
            <img src={resultUrl == "/placeholders/UntitledSLIC.jpg"? "/placeholders/UntitledSLIC.jpg": resultUrl} className="slic-result-image"/>
          </div>
        )}
      </div>
    </div>
  );
}

function TCAVScoring( ) {
    

    return (
        <>
            

            {/* {concepts.map((concept, index) => {
                if (concepts[0].length > 0) {
                    return(
                        concept.map((patch, index) => (
                            <img src={patch}/>
                        ))
                    )
                }
            })} */}

            ТУТА БУДЕТ ТКАВ СКОРИНГ ЁПТА
        </>
    )
}

function Carousel({ images }) {
  
  const extendedImages = [...images, ...images];

  return (
    <div className="carousel">
      <div className="carousel-track">
        {extendedImages.map((src, index) => (
          <div className="carousel-item" key={index}>
            <img src={src} loading="lazy" alt={`img-${index}`} />
          </div>
        ))}
      </div>
    </div>
  );
}

function ImagesHList({ images }) {

    return (
        <div className="images-h-list">
            <div className="images-container">
                {
                    images.map((src, index) => (
                        <div className="images-list-item" key={index}>
                            <img src={src} alt={`hlist-element-${index}`} />
                        </div>
                    ))
                }
            </div>
        </div>
    )
}

function Description({title, shortDesc, longDesc}) {
    const [isExpanded, setIsExpanded] = useState(false);

    return (<div className="description">
        <h2 className="description-title">
            {title}
        </h2>
        
        <p className="short-description">
            {shortDesc}
        </p>

        <div className="expand-button"
            onClick={() => setIsExpanded(!isExpanded)}
        >
            <div className="expand-message">{!isExpanded? "Expand v": "Shrink ^"}</div>
        </div>
        {isExpanded? 
            <p className="long-description highlited-section">
                {longDesc}
            </p>
        :
            <></>
        }
    </div>)
}

export default function MainPage() {
    const [concepts, setConcepts] = useState([])
    const [TCAVscores, setTCAVscores] = useState([])
    const [isTCAVLoading, setIsTCAVLoading] = useState(false)

    const [isClassChoice, setIsClassChoice] = useState(false)
    const [currentClass, setCurrentClass] = useState("tench")

    const [numConcepts, setNumConcepts] = useState(3)
    const [numImagesToAnalyze, setNumImagesToAnalyze] = useState(10)
    const [numCAVs, setNumCAVs] = useState(1)
    const [numExamples, setNumExamples] = useState(15)
    
    const classes = [
        "tench",
        "English springer",
        "cassette player"
    ]

    const class_index_mapping = {
        "tench": 0,
        "English springer": 1,
        "cassette player": 2
    }

    const handleTCAV = async (class_index, n_images, n_concepts, n_cavs, n_examples) => {
        setIsTCAVLoading(true);
        const queryString = new URLSearchParams({class_index, n_images, n_concepts, n_cavs, n_examples}).toString();
        console.log(queryString)

        const res = await fetch(`http://localhost:8000/extract_clusters_for_class?${queryString}`);

        setIsTCAVLoading(false);

        const data = await res.json();

        console.log(data);

        setConcepts(data[0]);
        setTCAVscores(data[1]);
        
    };

    return (
        <div className='main-page'>
            <h1 className='main-title'>
                From Features to Concepts: Demonstrating Automated concept-based Explainability in Action
            </h1>
            <p className="main-description highlited-section">
                We present a full demonstration of the Automated Concept-based Explainability (ACE) pipeline, starting from raw images and ending with the ranking of the discovered concepts using the TCAV score.
            </p>

            <div className="content-block">

                <Description 
                    title={"Why do we need ACE?"}
                    shortDesc={"ACE explains model decisions using abstract concepts beyond local, non-semantic saliency maps."}
                    longDesc={"ACE allows us to explain model decisions at a higher, more abstract level than pixel-based methods such as saliency maps, which highlight only local feature importance without capturing semantic meaning. This higher-level interpretability is essential for understanding what concepts a model has actually learned, improving trust in its decisions, and enabling more effective debugging and control over model behavior."}
                />

                <h2 className="raw-images-title">
                    Raw Images
                </h2>
                <p className="raw-images-description">
                    The raw images were taken from the Imagenette dataset, which is a subset of the larger ImageNet dataset. A convolutional neural network (CNN) was trained on this dataset, which will later allow us to analyze, manipulate, and cluster the learned concepts.
                </p>
            </div>

            <Carousel images={staticImages}/>

            <div className="content-block">
                <Description
                    title={"Simple Linear Iterative Clustering (SLIC)"}
                    shortDesc={"SLIC efficiently clusters pixels into compact superpixels using color-position space and tunable compactness."}
                    longDesc={"SLIC (Simple Linear Iterative Clustering) allows us to efficiently partition an image into a set of superpixels—compact, perceptually meaningful regions that group together pixels with similar color and spatial proximity. It operates by clustering pixels in a combined five-dimensional space (color + image coordinates), which makes it both fast and memory-efficient. We can control the approximate number of resulting segments, as well as their compactness. The compactness parameter balances color similarity against spatial proximity: higher values produce more regular, grid-like superpixels, while lower values allow segments to better adhere to object boundaries. It is important to note that SLIC does not explicitly “merge small segments together”; rather, it directly generates superpixels of roughly uniform size through its clustering process."}
                />
            </div>

            <SLICShowcase />

            <div className="content-block">
                <Description 
                    title={"K-Means Concept Clustering"}
                    shortDesc={"CNN embeddings of segments are clustered with K-Means to form semantically meaningful concepts."}
                    longDesc={"Using a CNN, we extract vector representations (embeddings) for each segment produced by SLIC. These embeddings capture high-level visual features of the segments. We then apply K-Means clustering to group similar embeddings together; each resulting cluster corresponds to a discovered concept, representing a recurring visual pattern in the data. In this way, concepts emerge as groups of semantically related image regions rather than predefined labels."}
                />
                <Description
                    title={"Clusters FIltering"}
                    shortDesc={"Outliers must be removed and clusters inspected to ensure coherent, meaningful learned concepts."}
                    longDesc={"To obtain meaningful clusters that correspond to the concepts of interest, it is important to filter out outliers and, ideally, manually inspect the resulting clusters—especially if the CNN is not sufficiently well trained. Outlier removal helps eliminate noisy or unrepresentative segments, while visual inspection ensures that the clusters align with coherent and interpretable concepts rather than artifacts or spurious patterns learned by the model."}
                />
            </div>

            <div className="concepts-hparams">
                <div style={{display:"flex", alignItems:"center", gap:"16px"}}>
                    Class to explain:
                    <div className="current-class"
                        onClick={() => setIsClassChoice(!isClassChoice)}>
                        {currentClass}
                    </div>
                </div>
                

                {isClassChoice? 
                    <div className="classes-dropdown">
                        {classes.map((item, index) => {
                            if (item != currentClass) {
                                return(
                            <div className="class-choice"
                                onClick={() => {
                                    setCurrentClass(item)
                                    setIsClassChoice(!isClassChoice)
                                    }}>
                                {item}
                            </div>)
                            }
                        })}
                    </div>
                    :
                    <>
                    </>
                }
                
                <div className="num-concepts">
                    Number of concepts to retrieve:
                    <Slider
                        defaultValue={3}
                        getAriaValueText={setNumConcepts}
                        valueLabelDisplay="off"
                        shiftStep={2}
                        step={1}
                        marks
                        min={1}
                        max={7}
                    />
                    {numConcepts}
                </div>
                
                <div className="num-images-to-analyze">
                    Number of Images to form ACE:
                    <Slider
                        defaultValue={10}
                        getAriaValueText={setNumImagesToAnalyze}
                        valueLabelDisplay="off"
                        shiftStep={5}
                        step={1}
                        marks
                        min={10}
                        max={50}
                    />
                    {numImagesToAnalyze}
                </div>
                <div className="num-cavs">
                    Number of CAVs:
                    <Slider
                        defaultValue={1}
                        getAriaValueText={setNumCAVs}
                        valueLabelDisplay="off"
                        shiftStep={4}
                        step={1}
                        marks
                        min={1}
                        max={20}
                    />
                    {numCAVs}
                </div>
                <div className="num-examples">
                    Number of Examples:
                    <Slider
                        defaultValue={15}
                        getAriaValueText={setNumExamples}
                        valueLabelDisplay="off"
                        shiftStep={5}
                        step={1}
                        marks
                        min={5}
                        max={35}
                    />
                    {numExamples}
                </div>
            </div>
            
            

            <button onClick={() => handleTCAV(class_index_mapping[currentClass], numImagesToAnalyze, numConcepts, numCAVs, numExamples)} className="tcav-processing special-button">
                Extract Concepts
            </button>

            {isTCAVLoading? <>Loading... Please Wait...</> : <></>}

            {TCAVscores.length > 0 ?
            <>  
                {concepts.map((item, index) => {
                    return(
                        <>
                            <h2 className="concept-title">Concept #{index} TCAV score: {TCAVscores[index]}</h2>
                            <ImagesHList images={item}/>
                        </>
                    )
                })}
            </>
            :
            <></>
            }

            <div className="content-block">
                <Description
                    title={"TCAV Scoring"}
                    shortDesc={"CAVs quantify concept influence by measuring prediction sensitivity along learned concept directions."}
                    longDesc={"After fully filtering the concepts, we need to test the dataset against these learned concepts. The idea behind testing with Concept Activation Vectors (CAVs) is as follows: for each concept, a linear classifier is trained to distinguish between examples of that concept and random counterexamples in the embedding space of the network. The normal vector of this classifier defines the CAV, which represents the direction corresponding to the concept in the model’s internal feature space. We then measure how sensitive the model’s predictions are to this concept by computing directional derivatives of the output with respect to the CAV. This allows us to quantify how strongly each concept influences the model’s decision, which is later summarized using TCAV scores."}
                />
            </div>

            <div className="summary highlited-section">
                <h2 className="summary-title">
                    Summary
                </h2>
                <p className="summary-description">
                    The ACE pipeline starts by segmenting images into superpixels (via SLIC), extracting feature embeddings for each segment using a CNN, and clustering these embeddings with K-Means to discover candidate visual concepts. After filtering outliers and refining clusters, Concept Activation Vectors (CAVs) are learned for each concept to represent directions in the model’s feature space. The influence of these concepts on model predictions is then quantified using TCAV scores, which measure how sensitive the model’s outputs are to each concept.

                    The results show which high-level, human-interpretable concepts the model relies on most for its decisions. High TCAV scores indicate that a concept has a strong positive influence on predictions, while low or inconsistent scores suggest weak or unreliable relevance. Overall, this provides insight into the model’s internal reasoning, helping identify both meaningful learned patterns and potential biases or artifacts.
                </p>
            </div>
        </div>
    )
}