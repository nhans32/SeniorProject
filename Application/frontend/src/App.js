import { Button, Typography, Tooltip } from '@mui/material';
import TextField from '@mui/material/TextField';
import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as ReChartsTooltip } from 'recharts';
import "./App.css";
import Select, { SelectChangeEvent } from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';

function App() {

  const NOHelp = "Note Onset: Tempo based on music note onset (good for percussive/non-percussive music)"
  const SEHelp = "Sound Energy: Tempo based on calculating sound energy blocks of audio amplitude (good for percussive music)"
  const LPFHelp = "Low Pass Filter: Tempo based on amplitude changes of low frequency audio elements (good for percussive music)"

  // Input state variables
  const [linkValue, setLinkValue] = useState('');
  const [selectedFileName, setSelectedFileName] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [tempoAlgo, setTempoAlgo] = useState("noa")
  const [tempoHelp, setTempoHelp] = useState(NOHelp)

  // Form tracking state variables
  const [submittedOnce, setSubmittedOnce] = useState(false);
  const [loading, setLoading] = useState(false);
  const [classifyDisabled, setClassifyDisabled] = useState(true);
  const [fileUploadDisabled, setFileUploadDisabled] = useState(false);
  const [linkDisabled, setLinkDisabled] = useState(false);

  // Output state variables
  const [classification, setClassification] = useState(null);

  // Error variables
  const [fetchError, setFetchError] = useState(false);

  const handleSubmit = () => {
    setSubmittedOnce(true);
    setLoading(true);
    setClassifyDisabled(true);

    var reqOptions = {};

    if (selectedFile != null && linkValue === '') { // using file
      console.log("Uploading File");
      console.log(selectedFileName)
      console.log(selectedFile)
      console.log(tempoAlgo)

      const data = new FormData();
      data.append('file', selectedFile);
      data.append('filename', selectedFileName);
      data.append('tempoAlgo', tempoAlgo);

      reqOptions = {
        method: 'POST',
        body: data
      }
    }
    else if (selectedFile == null && linkValue !== '') {  // using link
      console.log("Uploading Link");
      console.log(linkValue);
      console.log(tempoAlgo)

      reqOptions = {
        method: 'POST',
        headers: {"content-type": "application/json"},
        body: JSON.stringify({"link": linkValue, "tempoAlgo": tempoAlgo})
      }
    }
    else {
      console.log("Error: no file or link selected");
      return;
    }

    fetch("http://localhost:8000/classify", reqOptions)
    .then(response => {
      return response.json();
    })
    .then(data => {
      console.log(data);
      setLoading(false);
      setClassifyDisabled(false);
      setFetchError(false);
      setClassification(data);
    })
    .catch(error => {
      console.log("There was an error", error);
      setLoading(false);
      setClassifyDisabled(false);
      setFetchError(true);
      alert("There was an error with the request. Check that file/YouTube link is valid and please try again.");
    });
  }

  const handleLinkChange = (event) => {
    setLinkValue(event.target.value);
    if (event.target.value.length > 0) {
      setFileUploadDisabled(true);
      setClassifyDisabled(false);
    }
    else {
      setFileUploadDisabled(false);
      setClassifyDisabled(true);
    }
  }

  const handleFileChange = (event) => {
    setSelectedFileName(event.target.files[0].name);
    setSelectedFile(event.target.files[0]);
    setLinkDisabled(true); 
    setClassifyDisabled(false)
  }

  const handleClearFile = () => {
    setSelectedFileName('');
    setSelectedFile(null);
    setLinkDisabled(false);
    setClassifyDisabled(true);
  }

  const handleTempoAlgoChange = (event) => {
    setTempoAlgo(event.target.value)
    console.log(event.target.value)
    if (event.target.value === "noa") {
      setTempoHelp(NOHelp)
    }
    else if (event.target.value === "lfa") {
      setTempoHelp(LPFHelp)
    }
    else if (event.target.value === "sea") {
      setTempoHelp(SEHelp)
    }
  }

  return (
    <div>
      <div className="headerContainer">
        <Typography align="center" variant="h4">Music Genre & Mood Classifier</Typography>
        <Typography align="center" variant="h6">Senior Project by Nicholas Hansen (Advisor: Dr. Franz Kurfess)</Typography>
        <Typography align="center">Cal Poly, San Luis Obispo - Computer Science 2022</Typography>
        <Typography align="center" variant="h6">---</Typography>
      </div>

      <Typography align="center" variant="h6">Select File or YouTube Link</Typography>
      <Typography className="inputsInstr" align="center"><strong>Note:</strong> Only Songs with Length Under 10min Supported </Typography>
      <div className="inputsContainer">
        <div className="inputsItem">
          <Button onClick={handleClearFile} variant="text" disabled={fileUploadDisabled} style={{width:"5%", height:"100%", maxWidth:"5%", maxHeight:"100%"}}>x</Button>
          <Tooltip title={selectedFileName === '' ? "Select File" : selectedFileName}>
            <Button disabled={fileUploadDisabled} style={{width:"100%", height:"100%", maxWidth:"100%", maxHeight:"100%"}} variant="contained" component="label">
              {selectedFileName === '' ? "Select File" : selectedFileName.slice(0, 11) + "..."}
              <input accept="audio/*" type="file" hidden onChange={handleFileChange}/>
            </Button>
          </Tooltip>
        </div>

        <div className="inputsItem">
          <Typography align="center" variant="h6">or</Typography>
        </div>

        <div className="inputsItem">
          <Tooltip title={linkValue === '' ? "YouTube Link" : linkValue}>
            <TextField disabled={linkDisabled} style={{width:"100%", height:"100%"}} className="inputsItem" onChange={handleLinkChange} label="YouTube Link"/>
          </Tooltip>
        </div>
      </div>

      <Typography align="center" variant="h6">Tempo Detection Algorithm</Typography>
      <div className="inputsTempoAlgoContainer">
        <Select
          value={tempoAlgo}
          onChange={handleTempoAlgoChange}
        >
          <MenuItem value={"noa"}>
            Note Onset Algorithm
          </MenuItem>
          <MenuItem value={"lfa"}>
            Lowpass Filter Algorithm
          </MenuItem>
          <MenuItem value={"sea"}>
              Sound Energy Algorithm
          </MenuItem>
        </Select>
      </div>
      <Typography display="block"
        variant="caption"
        align="center" 
        className="tempoTip">
        {tempoHelp}
      </Typography>

      <div className="submitContainer">
        <Button disabled={classifyDisabled} style={{width:"10em", height:"100%"}} variant="contained" onClick={handleSubmit}>Classify</Button>
      </div>

      {loading && submittedOnce
        ?
        <div className="loadingContainer">
          <Typography align="center" variant="h6">Classification Information</Typography>
          <Typography align="center" variant="h6">---</Typography>
          <Typography align="center"><strong>Loading...</strong></Typography>
        </div>
        : (!loading && submittedOnce && !fetchError
          ? 
            <div className="classificationContainer">
                <Typography align="center" variant="h6">Classification Information</Typography>
                <Typography align="center" variant="h6">---</Typography>
                <Typography align="center"><strong>File Name/Song Title:</strong> {classification.song_title}</Typography>
                <Typography align="center"><strong>Predicted Genre:</strong> {classification.genre.prediction}</Typography>
                <Typography align="center"><strong>Predicted Mood:</strong> {classification.mood.prediction}</Typography>
                <Typography align="center"><strong>Predicted Tempo:</strong> {classification.tempo}</Typography>
                <div className="chartContainer">
                  <div className="chart">
                    <Typography align="center">Genre Probability Distribution</Typography>
                    <BarChart width={800} height={400} data={classification.genre.probabilities}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="genre" />
                        <YAxis domain={[0, 1]}/>
                        <ReChartsTooltip />
                        <Bar dataKey="probability" fill="#3f50b5" />
                    </BarChart>
                  </div>
                  <div className="chart">
                    <Typography align="center">Mood Probability Distribution</Typography>
                    <BarChart width={800} height={400} data={classification.mood.probabilities}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="mood" />
                        <YAxis domain={[0, 1]}/>
                        <ReChartsTooltip />
                        <Bar dataKey="probability" fill="#3f50b5" />
                    </BarChart>
                  </div>
                </div>
            </div> 
          : <div>
              <Typography align="center" variant="h6">Classification Information</Typography>
              <Typography align="center" variant="h6">---</Typography>
              <Typography align="center"><strong>*Nothing Submitted Yet*</strong></Typography>
            </div>
        )
      }
    </div>
  );
}

export default App;
