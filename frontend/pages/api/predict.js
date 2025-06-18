export default function handler(req, res) {
    // post request from res function
    if (req.method === 'POST') {
        const { age } = req.body; // data dari requestnya

        // Dummy Model
        let category = '';
        if (age < 18) category = 'Anak-Anak';
        else if (age < 60) category = 'Dewasa';
        else category = 'Lansia';
        
        // men json hasil category, dan kembalikan dengan status kode
        res.status(200).json({ result: category});
    } else {
        res.status(405).json({ error: 'Method not allowed'});
    }
}