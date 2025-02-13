import React, { useEffect, useState } from 'react';
import PunList from '../components/PunList'

function Evaluation() {
    const [headlines, setHeadlines] = useState([]);
    const [selectedHeadline, setSelectedHeadline] = useState(0);

    // Gets all available headlines
    useEffect(() => {
        const fetchHeadlines = async () => {
            const res = await fetch('http://localhost:5000/get_headlines');
            const data = await res.json();
            setHeadlines(data);
        }
        fetchHeadlines();
    }, []);

    const handleSelectChange = (e) => {
        setSelectedHeadline(e.target.value);
    }

    return (
        <div>
            <h2>Evaluate Headline</h2>
            <select value={selectedHeadline} onChange={handleSelectChange}>
                <option value="">Select a headline</option>
                {headlines.map((headline, index) => (
                    <option key={index} value={headline.headline_id}>{headline.headline}</option>
                ))}
            </select>
            <h3>Puns</h3>
            <PunList headline={selectedHeadline} />
        </div>
    );
}

export default Evaluation;
