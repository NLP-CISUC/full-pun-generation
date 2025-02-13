import React, { useEffect, useState } from 'react';
import Pun from './Pun';

function PunList({ headline }) {
    const [puns, setPuns] = useState([]);

    useEffect(() => {
        if (headline) {
            const fetchPuns = async () => {
                const res = await fetch(`http://localhost:5000/get_generated?id=${headline}`);
                const data = await res.json();
                setPuns(data);
            }
            fetchPuns();
        } else {
            setPuns([]);
        }
    }, [headline]);

    if (!headline) return <div>Selecione uma notícia</div>;

    return (
        <div>
            {puns.map((pun, index) => (
                <Pun key={index} text={pun.generated} />
            ))}
        </div>
    );
}

export default PunList;

