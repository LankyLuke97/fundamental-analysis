import { useCallback, useEffect, useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from './assets/vite.svg'
import heroImg from './assets/hero.png'
import './App.css'

const DATA_ENDPOINT = 'http://localhost:8000';
const VALID_API = ['transactions']

const App = () => {
    const [transactionItems, setTransactionItems] = useState([]);

    const fetchEndpoint = useCallback(async () => {
        try {
            const response = await fetch(`${DATA_ENDPOINT}/transactions`);
            const result = await response.json();
            setTransactionItems(result);
        } catch (error) {
            console.log(`Error while downloading from ${DATA_ENDPOINT}:\n`, error);
        }
    }, []);

    useEffect(() => {fetchEndpoint()}, [fetchEndpoint]);

    return (
        <>
            <List items={transactionItems} />
        </>
    );
};

const List = ({items, children}) => (
    <>
    <ul>
        {items.map(item => <li key={item.id}>
            <span>{item.ticker}</span>&nbsp;
            <span>{item.price}</span>&nbsp;
            <span>{item.quantity}</span>&nbsp;
            <span>{item.timestamp}</span>&nbsp;
        </li>)}
    </ul>
    </>
);

export default App
