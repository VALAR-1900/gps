function rms(){

    var time=0,increment=1/2500,i=1,delaytime=1000,dec=3,number=1,number_iterated=1,i_before=0;
    console.log(time,increment);
    const xlabels=[];
    const ytemps=[];
    const xlabels_rms_wear=[];
    const ytemps_rms=[];
    const ytemps_wear=[];
    var func=setInterval(async function()
    {
             if(Math.floor(time)%dec == 0 && Math.floor(time)!=0)
            {
                chartit_rms_wear();
            }

            await getdata();  

            const ctx = document.getElementById('chart').getContext('2d');
            const myChart = new Chart(ctx, {
            type: 'line',
            data: {
            label:'Time',
            labels: xlabels,
            datasets: [{
                label: 'Vibration',
                data: ytemps,
                fill: false,
                borderColor:'rgba(0, 0, 0, 1)',
                borderWidth: 1
            }]
            },
            options:{
                elements: {
                point:{
                    radius: 0
                }
                },
                animation:null,
                scales:{
                    xAxes: [{
                        gridLines: {
                            color: "rgba(0, 0, 0, 0)",
                        },
                        ticks: {
                            max: 70,
                            min: 0,
                            // scaleOverride: true,
                            // scaleSteps: 10,
                            // scaleStepWidth: 5,
                            // scaleStartValue: 0
                            beginAtZero: true,
                            callback: function (value) { Math.floor(value) },
                            stepSize: 1
                    }
                   }],
                   yAxes: [{
                     gridLines: {
                     color: "rgba(0, 0, 0, 0)",
                    }   
                    }]
                }
            }


                });
    i_before=i;
    i++;
    if(i==15)
        delaytime=0;
    if(i>41)
            check();
    }, delaytime);
    function check()
    {
        clearInterval(func);
        popup();
    }
    
    async function getdata() {
            const response= await fetch('static/RMS_MOTOR1/speed_' + i + '.csv');
            const data = await response.text();
            const table = data.split('\n').slice(1);
            table.forEach( row => {
                const columns= row.split(',');
                xlabels.push(time);
                ytemps.push(columns[0]);
                //console.log(time ,v_y);
                time+=increment;
            });
    }

    async function chartit_rms_wear()
    {   
            await getdata_rms_wear();
            number_iterated++;
            const ctx_rms= document.getElementById('chart_rms').getContext('2d');
            const myChart_rms= new Chart(ctx_rms, {
            type: 'line',
            data: {
            labels: xlabels_rms_wear,
            datasets: [{
                label: 'Rms',
                data: ytemps_rms,
                fill: false,
                borderColor:'rgba(0, 0, 0, 1)',
                borderWidth: 1
            }]
            },
            options:{
                elements: {
                point:{
                    radius: 0
                }
                },
                animation:null,
                scales:{
                    xAxes: [{
                        gridLines: {
                            color: "rgba(0, 0, 0, 0)",
                        },
                        ticks: {
                        autoSkip: true,
                        // tickLenghth: 20,
                        // tickPixelInterval: 20,
                        // maxTicksLimit: 1,
                        showScale: true,
                        scaleOverride: true,
                        scaleSteps: 10,
                        scaleStepWidth: 3,
                        scaleLineWidth: 5,
                        scaleStartValue: 0,
                        stepSize: 0.5
                    }
                    }],
                    yAxes: [{
                        gridLines: { 
                        color: "rgba(0, 0, 0, 0)",
                        },
                        ticks: {
                            max: 0.15,
                            min: 0
                        }   
                    }]
                }
            }
                });
    const ctx_wear= document.getElementById('chart_wear').getContext('2d');
    const myChart_wear= new Chart(ctx_wear, {
    type: 'line',
    data: {
    labels: xlabels_rms_wear,
    datasets: [{
    label: 'Wear',
    data: ytemps_wear,
    fill: false,
    borderColor:'rgba(0, 0, 0, 1)',
    borderWidth: 1
    }]
    },
    options:{
                elements: {
                point:{
                    radius: 0
                }
                },
                animation:null,
                scales:{
                    xAxes: [{
                        gridLines: {
                            color: "rgba(0, 0, 0, 0)",
                        },
                        ticks: {
                            autoSkip: true,
                            // tickLenghth: 20,
                            // tickPixelInterval: 20,
                            // maxTicksLimit: 1,
                            showScale: true,
                            scaleOverride: true,
                            scaleSteps: 15,
                            scaleStepWidth: 3,
                            scaleLineWidth: 5,
                            scaleStartValue: 0,
                            stepSize: 0.5
                    }
                    }],
                    yAxes: [{
                        gridLines: {
                        color: "rgba(0, 0, 0, 0)",
                        },
                        ticks:{
                            max: 0.8,
                            min: 0
                        }   
                    }]
                }
            }
    });  
    xlabels_rms_wear.length=0;
    ytemps_rms.length=0;
    ytemps_wear.length=0;     
    }
    async function getdata_rms_wear() {
    const response= await fetch('static/RMS_MOTOR1/RMS and wear.csv');
    const data = await response.text();
    const table = data.split('\n').slice(1);
    ytemps_wear.push(0);
    ytemps_rms.push(0);
    xlabels_rms_wear.push(0);
    var time_wear_rms=3;
    table.forEach( row => {
                const columns= row.split(',');
                const rms_data=columns[0];
                const wear_data=columns[1];
                const temp=number_iterated*3;
                xlabels_rms_wear.push(time_wear_rms);
                if(temp>=time_wear_rms)
                {
                    ytemps_wear.push(wear_data);
                    ytemps_rms.push(rms_data);
                }
                time_wear_rms+=dec;
            });
                
        console.log("RMS AND WEAR DATA");
        console.log(time,xlabels_rms_wear,ytemps_rms,ytemps_wear);
    }   

}
function popup()
{
    var answer=Math.random() * (47 - 45) + 40;
    swal("Failure Mode: Breakage, Component Failed at: " + answer, {
        buttons: {
            cancel: "cancel",
                catch: {
                    text: "Save",
                    value: "saved",
                },
            },
        })
        .then((value) => {
            switch (value) {
            case "saved":
                swal("Saved", "Result is downloaded!", "success");
                //window.location.replace("/static/download.xlsx");
                window.location.replace("/download");
                break;
            }
        });
}

function popup2()
{
    swal(" UNDER DEVELOPEMENT ");
}